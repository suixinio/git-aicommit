package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-git/go-git/v5"
	"github.com/go-git/go-git/v5/plumbing/object"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

// Config structure for DeepSeek API
type Config struct {
	DeepSeek DeepSeekConfig `mapstructure:"deepseek"`
}

type DeepSeekConfig struct {
	APIKey      string   `mapstructure:"api_key"`
	Temperature *float32 `mapstructure:"temperature"`
	Prompt      *string  `mapstructure:"prompt"`
}

// ChatMessage represents a message in the chat completion request
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents the request to DeepSeek API
type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature *float32      `json:"temperature,omitempty"`
}

// StreamResponseChunk represents a chunk of the streaming response
type StreamResponseChunk struct {
	Choices []StreamChoice `json:"choices"`
}

type StreamChoice struct {
	Delta DeltaMessage `json:"delta"`
}

type DeltaMessage struct {
	Content *string `json:"content,omitempty"`
}

var (
	applyFlag bool
	rootCmd   = &cobra.Command{
		Use:               "ai-commit",
		Short:             "Create an AI-generated commit",
		Long:              `Generate commit messages using AI and optionally create commits with those messages.`,
		DisableAutoGenTag: true,
		RunE:              runCommit,
	}
)

func init() {
	rootCmd.Flags().BoolVarP(&applyFlag, "apply", "a", false, "Apply the AI-generated message to the new commit")

	// Disable completion command
	rootCmd.CompletionOptions.DisableDefaultCmd = true
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func runCommit(cmd *cobra.Command, args []string) error {
	// Load configuration
	config, err := loadConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	if config.DeepSeek.APIKey == "" {
		return fmt.Errorf("error: No DeepSeek API key found. Please set your API key in the config file")
	}

	// Get staged changes
	changes, err := getStagedChanges()
	if err != nil {
		return fmt.Errorf("failed to get staged changes: %w", err)
	}

	// Build prompt messages
	messages := buildPromptMessages(changes, config.DeepSeek.Prompt)

	// Print banner
	printBanner("AI Suggested Commit Message")

	// Stream commit message
	fullMessage, err := streamCommitMessage(config.DeepSeek.APIKey, messages, config.DeepSeek.Temperature)
	if err != nil {
		return fmt.Errorf("failed to generate commit message: %w", err)
	}

	// Apply commit if requested
	if applyFlag {
		commitID, err := createCommit(strings.TrimSpace(fullMessage))
		if err != nil {
			return fmt.Errorf("failed to create commit: %w", err)
		}
		printBanner("✅ Commit Successful")
		fmt.Printf("Commit ID: %s\n\n", commitID)
	}

	return nil
}

func loadConfig() (*Config, error) {
	// Set config name and type
	viper.SetConfigName("config")
	viper.SetConfigType("toml")

	// Add config path - only from user home directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get user home directory: %w", err)
	}
	viper.AddConfigPath(filepath.Join(homeDir, ".config", "git-aicommit"))

	// Try to read config
	err = viper.ReadInConfig()
	if err != nil {
		// If config doesn't exist, create a default one
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			if err := createDefaultConfig(); err != nil {
				return nil, err
			}
			// Try reading again
			if err := viper.ReadInConfig(); err != nil {
				return nil, err
			}
		} else {
			return nil, err
		}
	}

	var config Config
	if err := viper.Unmarshal(&config); err != nil {
		return nil, err
	}

	return &config, nil
}

func createDefaultConfig() error {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	configDir := filepath.Join(homeDir, ".config", "git-aicommit")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return err
	}

	configPath := filepath.Join(configDir, "config.toml")
	configContent := `
# Git-Aicommit Configuration File
# This file contains configuration settings for the git-aicommit CLI tool

[deepseek]
# DeepSeek API key for AI-powered commit message generation
# Get your API key from: https://platform.deepseek.com/
api_key = ""

# Temperature setting for AI text generation (0.0 to 2.0)
# Lower values (e.g., 0.1) make output more focused and deterministic
# Higher values (e.g., 1.5) make output more random and creative
# Default: 0.7 provides a good balance
temperature = 0.7

# Custom prompt for commit message generation (optional)
# Leave empty to use the default prompt
# The prompt should instruct the AI how to format commit messages
prompt = """
You are an AI commit message assistant.

Please generate a commit message with the following format:
1. Title (one short sentence, 50-72 characters max).
2. A clear bullet-point list of changes (start each line with "- ").
3. Each line, including bullets, should be under 100 characters.
4. Keep it concise, consistent, and professional.

Example:

Improve error handling in user authentication

- Add detailed error messages for login failures
- Handle timeout errors gracefully
- Refactor error propagation logic for clarity
"""
`

	return os.WriteFile(configPath, []byte(configContent), 0644)
}

func getStagedChanges() (string, error) {
	// Use git command to get staged changes
	cmd := exec.Command("git", "diff", "--cached")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	if len(output) == 0 {
		return "", fmt.Errorf("no staged changes found")
	}

	return string(output), nil
}

func buildPromptMessages(changes string, promptOpt *string) []ChatMessage {
	prompt := `
You are an AI commit message assistant.

Please generate a commit message with the following format:
1. Title (one short sentence, 50-72 characters max).
2. A clear bullet-point list of changes (start each line with "- ").
3. Each line, including bullets, should be under 100 characters.
4. Keep it concise, consistent, and professional.

Example:

Improve error handling in user authentication

- Add detailed error messages for login failures
- Handle timeout errors gracefully
- Refactor error propagation logic for clarity
`

	if promptOpt != nil && *promptOpt != "" {
		prompt = *promptOpt
	}

	return []ChatMessage{
		{
			Role:    "system",
			Content: prompt,
		},
		{
			Role:    "user",
			Content: fmt.Sprintf("Here are my current Git changes:\n%s", changes),
		},
	}
}

func printBanner(title string) {
	maxWidth := 100
	minWidth := 60
	padding := 4
	rawWidth := len(title) + padding*2
	totalWidth := maxWidth
	if rawWidth > minWidth && rawWidth < maxWidth {
		totalWidth = rawWidth
	} else if rawWidth <= minWidth {
		totalWidth = minWidth
	}

	bannerLine := strings.Repeat("=", totalWidth)
	titlePadding := (totalWidth - len(title)) / 2

	fmt.Println(bannerLine)
	fmt.Printf("%s%s%s\n",
		strings.Repeat(" ", titlePadding),
		title,
		strings.Repeat(" ", totalWidth-titlePadding-len(title)))
	fmt.Println(bannerLine)
}

func streamCommitMessage(apiKey string, messages []ChatMessage, temperature *float32) (string, error) {
	requestBody := ChatRequest{
		Model:       "deepseek-chat",
		Messages:    messages,
		Stream:      true,
		Temperature: temperature,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", "https://api.deepseek.com/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("API failed: %s", string(body))
	}

	var fullMessage strings.Builder
	var currentLine strings.Builder

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "data:") && line != "data: [DONE]" {
			jsonStr := strings.TrimSpace(line[5:])
			if jsonStr == "" {
				continue
			}

			var chunk StreamResponseChunk
			if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
				continue
			}

			for _, choice := range chunk.Choices {
				if choice.Delta.Content != nil {
					content := *choice.Delta.Content
					for _, ch := range content {
						currentLine.WriteRune(ch)
						if ch == '\n' {
							line := currentLine.String()
							fmt.Printf("| %s", strings.TrimSuffix(line, "\n"))
							fmt.Println()
							fullMessage.WriteString(line)
							currentLine.Reset()
						}
					}
				}
			}
		}
	}

	// Print any remaining content
	if currentLine.Len() > 0 {
		line := currentLine.String()
		fmt.Printf("| %s\n", strings.TrimSpace(line))
		fullMessage.WriteString(line)
	}

	if err := scanner.Err(); err != nil {
		return "", err
	}

	return fullMessage.String(), nil
}

func createCommit(message string) (string, error) {
	// Get current working directory
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}

	// Open git repository (search upwards from current directory)
	repo, err := git.PlainOpenWithOptions(wd, &git.PlainOpenOptions{
		DetectDotGit: true,
	})
	if err != nil {
		return "", err
	}

	// Get the working tree
	worktree, err := repo.Worktree()
	if err != nil {
		return "", err
	}

	// Create commit
	commit, err := worktree.Commit(message, &git.CommitOptions{
		Author: &object.Signature{
			Name:  getGitConfig("user.name"),
			Email: getGitConfig("user.email"),
			When:  time.Now(),
		},
	})
	if err != nil {
		return "", err
	}

	return commit.String(), nil
}

func getGitConfig(key string) string {
	cmd := exec.Command("git", "config", key)
	output, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(output))
}
