for FILE in $(find . -name "*.py"); do
    yapf -i $FILE --style="{based_on_style: pep8; DEDENT_CLOSING_BRACKETS: True, SPLIT_BEFORE_NAMED_ASSIGNS: True}";
done
