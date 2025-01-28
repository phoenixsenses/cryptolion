#!/bin/bash

# Variables - Update these with your details
GITHUB_USERNAME="Marmarex"
REPO_NAME="binance-testnet-bot"
BRANCH_NAME="main"

# Step 1: Remove existing .git directory
echo "Cleaning up old Git history..."
rm -rf .git
echo "Old Git history removed."

# Step 2: Initialize a new Git repository
echo "Initializing a new Git repository..."
git init

# Step 3: Add all project files to Git
echo "Adding all files to the new repository..."
git add .

# Step 4: Commit the files
echo "Creating an initial commit..."
git commit -m "Initial commit"

# Step 5: Add remote repository
REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo "Setting up remote repository: $REMOTE_URL"
git remote add origin $REMOTE_URL

# Step 6: Rename branch to main
echo "Renaming branch to '$BRANCH_NAME'..."
git branch -M $BRANCH_NAME

# Step 7: Push to GitHub
echo "Pushing code to GitHub..."
git push -u origin $BRANCH_NAME

# Done
echo "Repository successfully set up and pushed to: $REMOTE_URL"
