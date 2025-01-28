#!/bin/bash

# Check if .git exists
if [ ! -d ".git" ]; then
    echo "The .git directory does not exist in the current folder."
    exit 1
fi

echo "Found .git directory. Proceeding with deletion..."

# Step 1: Check ownership
echo "Checking ownership of .git directory..."
ls -ld .git

# Step 2: Change ownership to the current user if needed
echo "Changing ownership of .git directory to current user..."
sudo chown -R $USER:$USER .git

# Step 3: Modify permissions to allow write access
echo "Setting write permissions for .git directory..."
chmod -R u+w .git

# Step 4: Attempt to delete the .git directory
echo "Attempting to delete the .git directory..."
sudo rm -rf .git

# Step 5: Verify deletion
if [ ! -d ".git" ]; then
    echo ".git directory successfully deleted."
else
    echo "Failed to delete .git directory. Please check manually."
fi
