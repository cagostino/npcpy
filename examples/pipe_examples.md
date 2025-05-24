# Pipe file content to NPC sample command
cat data.json | npc sample "Summarize this JSON data"

# Pipe command output to NPC
ls -la | npc sample "Explain what these files are"

# Use grep to filter logs and have NPC analyze them
grep ERROR /var/log/application.log | npc sample "What are the common error patterns?"

# Use curl to get API response and analyze with NPC
curl -s https://api.example.com/data | npc sample "Analyze this API response"

# Create a multi-line prompt using heredoc
cat << EOF | npc sample
I need to understand how to structure a React application.
What are the best practices for component organization?
Should I use Redux or Context API for state management?
EOF


# Chain NPC commands using xargs
npc search "machine learning algorithms" | xargs -I {} npc sample "Explain {} in detail"

# Use output from one NPC command as input to another
npc sample "Generate 5 test cases" | npc sample "Convert these test cases to JavaScript code"

# Use NPC to generate code and then analyze it
npc sample "Write a Python sorting algorithm" | npc sample "Review this code for efficiency"

# Generate image description and then create image
npc sample "Describe a futuristic cityscape" | xargs npc vixynt


# Save NPC output to a file
npc sample "Write a Python script to process CSV files" > process_csv.py

# Count words in NPC response
npc sample "Write a short essay about AI" | wc -w

# Format NPC output using jq (if JSON output is enabled)
npc search "cryptocurrency news" --format=json | jq '.results[0]'

# Use NPC output as input to another tool
npc sample "Generate a list of SQL commands" | sqlite3 mydatabase.db

# Filter NPC response
npc sample "List 20 programming languages with their use cases" | grep -i "python"

# Send NPC output to clipboard
npc sample "Write a shell script to backup files" | xclip -selection clipboard

# Generate code and make it executable
npc sample "Write a bash script to organize files by extension" > organize.sh && chmod +x organize.sh

# Save NPC image generation to specific file
npc vixynt "A cyberpunk city" | convert - cyberpunk_city.png