#!/bin/sh
export FLASK_APP=$1
export FLASK_DEBUG=1
flask run
