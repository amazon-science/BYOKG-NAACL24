#!/usr/bin/bash

# Get Freebase KG from https://github.com/dki-lab/Freebase-Setup/
wget https://www.dropbox.com/s/q38g0fwx1a3lz8q/virtuoso_db.zip -O freebase_db.zip
unzip -d ./ freebase_db.zip
mv ./virtuoso_db/virtuoso.db ./virtuoso/virtuoso.db
rm -rf ./virtuoso_db
rm freebase_db.zip
