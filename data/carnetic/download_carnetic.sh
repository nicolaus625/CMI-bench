# you need to request the access on zenodo
# Step 1: Open Chrome DevTools
# Go to Zenodo and log in (if required).
# Press F12 (or Ctrl + Shift + I / Cmd + Option + I on Mac) to open DevTools.
# Click on the "Network" tab.
# Step 2: Start the Download
# Click the download link in your browser (without closing DevTools).
# Look for a request that starts with CMR_full_dataset_1.0.zip?download=1 in the Network tab.
# Click on that request.
# Step 3: Get the Cookie
# In the request details, go to the "Headers" tab.
# Scroll down to "Request Headers" and find "Cookie".
# Copy the entire cookie string (it will look like session=XXXXX; _ga=XXXXX; ...).
wget --header="Cookie: xxx" "https://zenodo.org/records/1264394/files/CMR_full_dataset_1.0.zip?download=1"ls 
unzip CMR_full_dataset_1.0.zip?download=1
rm CMR_full_dataset_1.0.zip?download=1