dirname=NEW_DIR
mkdir $dirname
find tuun -name '*.py' -exec cp --parents \{\} $dirname \;
