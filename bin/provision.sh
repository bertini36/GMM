#!/bin/bash
echo "Installing virtualenvwrapper in case it is not installed"
sudo pip install virtualenvwrapper
echo "Configuring environment variables"
source bin/start-env.sh
echo "Copying files"
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating virtual environment $VIRTUAL_ENV"
    #echo "Really making"
    mkvirtualenv $PROJECT
fi
echo "Installing packages in virtual environment"
pip install -r requirements.txt
