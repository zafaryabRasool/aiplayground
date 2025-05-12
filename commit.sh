#!/bin/bash

# Run 'git add .' to add all changes to staging
git add .

read -p "Enter commit message: " commit_message

# Run 'git commit -m "message"' to commit changes
git commit -m "$commit_message"

# Run 'git push' to push changes to remote repository
git push
