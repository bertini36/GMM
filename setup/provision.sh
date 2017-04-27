#!/usr/bin/env bash

# Execution: ./provision.sh

echo "Configuring environment variables"
export PROJECT=GMM2
export WORKON_HOME=~/.virtualenvs

echo "Installing virtualenv and virtualenvwrapper in case it is not installed"
sudo pip install virtualenv
sudo pip install virtualenvwrapper

echo "Creating virtual environment"
. `which virtualenvwrapper.sh`
mkvirtualenv $PROJECT
workon $PROJECT

echo "Installing packages in virtual environment"
pip install -r requirements.txt
