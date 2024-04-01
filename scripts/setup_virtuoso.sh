#!/usr/bin/bash

# Download virtuoso
gdown 1EFVs0BroyivrU5PsHQJNDw2q3ULAkLK6 -O virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
# Alternate link:
# wget https://sourceforge.net/projects/virtuoso/files/virtuoso/7.2.5/virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz/download -O virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
tar -xvf virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz -C ./virtuoso/
rm virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
