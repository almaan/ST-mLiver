#!/bin/bash
set -e

REPO=$1

if [ ! -d tmpdir ]; then
    mkdir tmpdir
else
    rm tmpdir
    mkdir tmpdir
fi
echo "Fetching data from >> $REPO"

curl $REPO --output tmpdir/data.zip

unzip tmpdir/data.zip
rm tmpdir/data.zip
unzip tmpdir/Hepaquery_data.zip
echo "Moving files to data dir"
cp -r tmpdir/Hepaquery_data/* data/
rm -r tmpdir
echo "Data Fetching Completed"


