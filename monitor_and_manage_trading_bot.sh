#!/bin/bash

# =============================================================================
# Script: monitor_and_manage_trading_bot.sh
# Description: Monitors the trading bot and ensures it runs continuously.
#              Restarts the bot if it crashes and sends email notifications
#              in case of critical errors.
# Author: ChatGPT
# Date: 2025-01-30
# =============================================================================

# Configuration
BOT_SCRIPT="trading_bot.py"
VENV_PATH=".venv/bin/activate"
LOG_FILE="bot_output.log"
MAX_RESTARTS=5
RESTART_INTERVAL=300  # 5 minutes
EMAIL_RECIPIENT="your_email@gmail.com"  # Update this after password change
EMAIL_SUBJECT="Trading Bot Alert: Crash Detected"

# Initialize counters
restart_count=0
restart_times=()

# Function to send email notifications
send_email() {
    local message="$1"
    echo -e "$message" | mail -s "$EMAIL_SUBJECT" "$EMAIL_RECIPIENT"
}

# Function to check if the bot is running
is_bot_running() {
    pgrep -f "$BOT_SCRIPT" > /dev/null 2>&1
    return $?
}

# Function to start the trading bot
start_bot() {
    echo "$(date) - Starting the trading bot..."
    # Activate virtual environment
    if [ -f "$VENV_PATH" ]; then
        source "$VENV_PATH"
    else
        echo "Virtual environment activation script not found at $VENV_PATH."
        send_email "Trading Bot failed to start because the virtual environment was not found."
        exit 1
    fi
    # Run the trading bot and redirect output to log file
    python "$BOT_SCRIPT" > "$LOG_FILE" 2>&1 &
    BOT_PID=$!
    echo "$(date) - Trading bot started with PID: $BOT_PID"
}

# Function to monitor the log file for critical errors
monitor_log() {
    # Look for critical error keywords
    grep -Ei "(critical|error|exception)" "$LOG_FILE" | tail -n 10
}

# Function to clean up restart times outside the restart interval
cleanup_restart_times() {
    local current_time=$(date +%s)
    local interval_start=$((current_time - RESTART_INTERVAL))
    # Keep only restart times within the interval
    restart_times=($(for t in "${restart_times[@]}"; do
        if [ "$t" -ge "$interval_start" ]; then
            echo "$t"
        fi
    done))
}

# Main monitoring loop
while true; do
    if is_bot_running; then
        # Bot is running, check for critical errors
        critical_errors=$(monitor_log)
        if [[ ! -z "$critical_errors" ]]; then
            echo "$(date) - Critical errors detected in the log:"
            echo "$critical_errors"
            send_email "Critical errors detected in the trading bot:\n\n$critical_errors"
            # Restart the bot if critical errors are detected
            echo "$(date) - Restarting the trading bot due to critical errors."
            # Kill only the specific bot process
            BOT_PID=$(pgrep -f "$BOT_SCRIPT")
            if [ ! -z "$BOT_PID" ]; then
                kill -9 $BOT_PID
                echo "$(date) - Killed trading bot PID: $BOT_PID"
            fi
            start_bot
            # Record the restart time
            restart_times+=($(date +%s))
            restart_count=$((restart_count + 1))
            cleanup_restart_times
            # Check if restart count exceeds maximum within the interval
            if [[ "${#restart_times[@]}" -ge "$MAX_RESTARTS" ]]; then
                echo "$(date) - Trading bot has crashed $MAX_RESTARTS times within the last $((RESTART_INTERVAL / 60)) minutes."
                send_email "Trading Bot has crashed $MAX_RESTARTS times within the last $((RESTART_INTERVAL / 60)) minutes. Manual intervention required."
                exit 1
            fi
        fi
    else
        # Bot is not running, attempt to start it
        echo "$(date) - Trading bot is not running. Attempting to start."
        start_bot
        # Record the restart time
        restart_times+=($(date +%s))
        restart_count=$((restart_count + 1))
        cleanup_restart_times
        # Check if restart count exceeds maximum within the interval
        if [[ "${#restart_times[@]}" -ge "$MAX_RESTARTS" ]]; then
            echo "$(date) - Trading bot has crashed $MAX_RESTARTS times within the last $((RESTART_INTERVAL / 60)) minutes."
            send_email "Trading Bot has crashed $MAX_RESTARTS times within the last $((RESTART_INTERVAL / 60)) minutes. Manual intervention required."
            exit 1
        fi
    fi

    # Wait before the next check
    sleep "$RESTART_INTERVAL"
done
