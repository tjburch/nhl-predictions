#!/bin/bash

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Users/tburch/Documents/github/nhl-hierarchical/nhl-env/bin
PROJECT_PATH="/Users/tburch/Documents/github/nhl-hierarchical"
TIMESTAMP=$(date +%Y-%m-%d)
FILE_PATH="data/prepped/${TIMESTAMP}.pq"
MATCHUP_FILE="data/prepped/matchups_${TIMESTAMP}.pq"
LOG_PATH="logs/${TIMESTAMP}.txt"

echo "Starting cron job" >> ${PROJECT_PATH}/${LOG_PATH}

echo "Begin python wrapper" >> ${PROJECT_PATH}/${LOG_PATH}
${PROJECT_PATH}/nhl-env/bin/python ${PROJECT_PATH}/wrapper.py
cd ${PROJECT_PATH} >> ${PROJECT_PATH}/${LOG_PATH} 2>&1

echo "Start git" >> ${PROJECT_PATH}/${LOG_PATH}
git add ${FILE_PATH} >> ${PROJECT_PATH}/${LOG_PATH} 2>&1
echo "${FILE_PATH} added" >> ${PROJECT_PATH}/${LOG_PATH}
# Check if the matchups file exists
if [ -f ${PROJECT_PATH}/${MATCHUP_FILE} ]; then
    git add ${MATCHUP_FILE} >> ${PROJECT_PATH}/${LOG_PATH} 2>&1
    echo "${MATCHUP_FILE} added" >> ${PROJECT_PATH}/${LOG_PATH}
else 
    echo "no matchups today" >> ${PROJECT_PATH}/${LOG_PATH}
fi

# Commit the changes
git commit -m "Add prepped data for ${TIMESTAMP}" >> ${PROJECT_PATH}/${LOG_PATH} 2>&1
echo "Committed with git" >> ${PROJECT_PATH}/${LOG_PATH}

# Push to Heroku
if git push heroku main >> ${PROJECT_PATH}/${LOG_PATH} 2>&1; then
    echo "Finished pushing to Heroku at $(date)" >> ${PROJECT_PATH}/${LOG_PATH}
else
    echo "Failed to push to Heroku at $(date)" >> ${PROJECT_PATH}/${LOG_PATH}
fi
