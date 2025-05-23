% Helper predicate for numbers 0-19
num_word(0, 'zero').
num_word(1, 'one').
num_word(2, 'two').
num_word(3, 'three').
num_word(4, 'four').
num_word(5, 'five').
num_word(6, 'six').
num_word(7, 'seven').
num_word(8, 'eight').
num_word(9, 'nine').
num_word(10, 'ten').
num_word(11, 'eleven').
num_word(12, 'twelve').
num_word(13, 'thirteen').
num_word(14, 'fourteen').
num_word(15, 'fifteen').
num_word(16, 'sixteen').
num_word(17, 'seventeen').
num_word(18, 'eighteen').
num_word(19, 'nineteen').

% Helper predicate for tens
tens_word(2, 'twenty').
tens_word(3, 'thirty').
tens_word(4, 'forty').
tens_word(5, 'fifty').
tens_word(6, 'sixty').
tens_word(7, 'seventy').
tens_word(8, 'eighty').
tens_word(9, 'ninety').

% Helper predicate for numbers less than 100
to_words_lt100(N, WordsAtom) :-
    N < 20,
    num_word(N, WordsAtom).
to_words_lt100(N, WordsAtom) :-
    N >= 20,
    Tens is N // 10,
    Units is N mod 10,
    tens_word(Tens, TensWord),
    (   Units =:= 0 ->
        WordsAtom = TensWord
    ;   num_word(Units, UnitWord),
        atomic_list_concat([TensWord, UnitWord], ' ', WordsAtom)
    ).

% Helper predicate for numbers less than 1000 (but >= 100)
to_words_lt1000(N, WordsAtom) :-
    Hundreds is N // 100,
    Remainder is N mod 100,
    num_word(Hundreds, HundredWord),
    (   Remainder =:= 0 ->
        atomic_list_concat([HundredWord, 'hundred'], ' ', WordsAtom)
    ;   to_words_lt100(Remainder, RemainderWords),
        atomic_list_concat([HundredWord, 'hundred', 'and', RemainderWords], ' ', WordsAtom)
    ).

% Predicate to get the word atom (internal use)
get_words_atom(N, WordsAtom) :-
    integer(N), N >= 0, N =< 1000, % Input validation
    (   N =:= 0 ->
        num_word(0, WordsAtom)
    ;   N =:= 1000 ->
        WordsAtom = 'one thousand'
    ;   N < 100 ->
        to_words_lt100(N, WordsAtom)
    ;   N < 1000 -> % This implies N >= 100
        to_words_lt1000(N, WordsAtom)
    ).

% Fallback for get_words_atom/2 for out of range or non-integer input
get_words_atom(N, 'Input out of range (0-1000)') :-
    \+ (integer(N), N >= 0, N =< 1000).

% Main predicate to_words/1 (prints the words)
to_words(N) :-
    get_words_atom(N, WordsAtom),
    write(WordsAtom),
    nl.