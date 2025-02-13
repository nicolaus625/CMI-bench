IF you use this test-set, please cite:

U. Marchand, G. Peeters. “Swing Ratio Estimation” in Proc. of the 18th Int. Conference on Digital Audio Effects (DAFx-15), December 2015.



-----------------------------------
The GTZAN-Rhythm test-set contains:
- GTZANindex.txt
- generate.py
- stats.csv
- xml folder containing 'swing', 'no_swing' and 'ternary' subfolders
- jams folder


--------------
GTZANindex.txt
--------------
This text file contains all the track informations compiled by Bob Sturm.

-----------
generate.py
-----------
This python script shows an example of how to read .xml annotation files.
It generates all the .jams files and stats.csv

---------
stats.csv
---------
It contains high-level annotations, compiled from the .xml files.
It contains a row for each track, with its filename, artist, title, tempo, swing, meter, ...

----------
xml folder
----------
It contains all the annotations in .xml files.
The folder 'swing' contains all the tracks that have swing.
The folder 'ternary' and 'no_swing' contains all the tracks that don't have swing.
But, as a ternary track can be seen as a track with a swing ratio of 2, we choose to separate ternary tracks from the other.
It could be argued that some tracks could be swinging AND ternary. In practice, we followed this protocol:
- If a track has swing, classify it in the 'swing' folder
- Then, if a track is ternary (beat is divided in 3 tatum), put it in the 'ternary' folder.
- Else, put the track in 'no_swing' folder.

-----------
jams folder
-----------
It contains all the informations of .xml files, and stats.csv, saved under another relatively used format: the JAMS format.
