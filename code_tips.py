# coding: utf-8


#  10-xxx: grammar problem
#  11-xxx: var type
#  12-xxx: object
#  13-xxx: ...


# cook-1001
#   grammar problem
#   notation: disusage []
#   do not use [] for fun,
#   do not use () for list, ruple, array, ..., and other sequence type var
def use_correct_brackets():
    a = [1, 2, 3]
    # error a(1) == 1


# cook-1002
#    immutable and hashable
#       1. All of Python's immutable build-in objects is hashable, for exp. : number, str, function, tuple
#       2. Hashable object has method: __hash__, __eq__, they use hash value to compare equal
#       3. Hashable object can be used as key in dict and as element in set, that need hash value
#       4. User-defined class object is hashble by default
#       5. Mutable container is not immutable and hashable
#    Hashable: can set up a fixed relation between id and value
#    the elements may be mutable in list, so list is not hashable
#    but type var list is not immutable and hashable
def hash_test():
    print(hash(list), hash(set), hash(dict))
