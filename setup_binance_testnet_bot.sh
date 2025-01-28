#!/bin/bash

# Configurations
GITHUB_USERNAME="Marmarex"
REPO_NAME="binance-testnet-bot"
BRANCH_NAME="main"

# Step 1: Clean up old Git history
echo "Cleaning up old Git history..."
rm -rf .git
echo "Old Git history removed."

# Step 2: Initialize a new Git repository
echo "Initializing a new Git repository..."
git init

# Step 3: Add a .gitignore file
echo "Adding a .gitignore file..."
cat <<EOL > .gitignore
# Ignore Python cache
__pycache__/
*.pyc
*.pyo

# Ignore environment files
.env
EOL

# Step 4: Add files and commit
echo "Adding all files to Git..."
git add .
echo "Creating initial commit..."
git commit -m "Initial commit"

# Step 5: Create a .env file for Binance Testnet credentials
echo "Creating .env file for Binance Testnet credentials..."
cat <<EOL > .env
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET_KEY=your_testnet_secret_key
EOL
echo ".env file created. Remember to replace 'your_testnet_api_key' and 'your_testnet_secret_key' with actual credentials!"

# Step 6: Add the remote repository
REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo "Adding remote repository: $REMOTE_URL"
git remote add origin $REMOTE_URL

# Step 7: Rename branch to main
echo "Renaming branch to '$BRANCH_NAME'..."
git branch -M $BRANCH_NAME

# Step 8: Push code to GitHub
echo "Pushing code to GitHub..."
git push -u origin $BRANCH_NAME
echo "Code successfully pushed to GitHub: $REMOTE_URL"

# Step 9: Guide for GitHub Secrets
echo -e "\nTo secure your Binance API keys, follow these steps:"
echo "1. Go to your GitHub repository: $REMOTE_URL"
echo "2. Navigate to Settings > Secrets and variables > Actions."
echo "3. Add the following secrets:"
echo "   - BINANCE_API_KEY: Your Binance Testnet API Key"
echo "   - BINANCE_SECRET_KEY: Your Binance Testnet Secret Key"
echo "This ensures your keys remain secure during CI/CD."

# Final message
echo -e "\nSetup complete! Your repository is now connected to GitHub, and a .env file has been created locally for your Binance Testnet credentials."
