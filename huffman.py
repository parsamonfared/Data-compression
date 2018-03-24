"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict(int,int)

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}

    for item in text:
        if not item in freq_dict:
            freq_dict[item] = 1
        else:
            freq_dict[item] = freq_dict[item] + 1
    return freq_dict
            
    
def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    # sorted the dictionary keys to list l in terms of it's key's values
    temp = [] #[0] = Frequency, [1] = HuffmanNode w/ Symbol
    for key in freq_dict:
        temp.append([freq_dict[key], HuffmanNode(key)])

   
    if len(temp) == 1:
        return HuffmanNode(temp.popitem()[0])
    
    while len(temp) > 1:
        temp = sorted(temp) #List of all of frequencies in the dictionary
        left = temp.pop(0)
        right = temp.pop(0)
        temp.append([left[0] + right[0], HuffmanNode(left = left[1], right = right[1])])
    return temp[0][1] #Returns the root node of the Huffman tree (all others have been combined)
                

def get_codes(tree):
    """ Return a dict mapping symbols from Huffman tree to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    return get_binary(tree, '', {})
    
def get_binary(tree, code, d = {}):
    
    if tree == None:        
        return {}
        
    elif tree.left == tree.right and tree.left != None:     
        return {tree.symbol: '1'}
    
    if tree.left != None:
        get_binary(tree.left, code + '0', d)
        
    if tree.right != None:
        get_binary(tree.right, code + '1', d)
        
    if tree.is_leaf():
        d[tree.symbol] = code
    
    return d

def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    do_number_nodes(tree, [0])
    
def do_number_nodes(tree, cur_number):
    
    if tree.left != None: #Left exists
        do_number_nodes(tree.left, cur_number)
    
    if tree.right != None: #Right exists
        do_number_nodes(tree.right, cur_number)
        
    if not tree.is_leaf(): #Ignores leaves
        tree.number = cur_number[0]
        cur_number[0] += 1
        return    
    
def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    d = get_codes(tree)
    freq_num = 0
    binary_num_sizes = 0
    
    for key in d:
        freq_num += freq_dict[key]
        binary_num_sizes += (len(d[key])* freq_dict[key])
    
    return binary_num_sizes/freq_num
    
        
def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mapping from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    bits = ''
    byte = []
    for item in text:
        bits += codes[item]
        if len(bits) >= 8:
            byte.append(bits_to_byte(bits[:8]))
            bits = bits[8:]
            
    if len(bits) != 0:
        byte.append(bits_to_byte(bits))
        
    return bytes(byte)
    
    
        
    


def tree_to_bytes(tree):
    """ Return a bytes representation of the Huffman tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    return bytes(get_byte(tree))

def get_byte(tree):
    
    if tree.is_leaf():
        return []
    if tree.left != None:
        if tree.left.is_leaf():
            first_byte = 0
            second_byte = tree.left.symbol
        
        else:
            first_byte = 1
            second_byte = tree.left.number

    if tree.right != None:   
        if tree.right.is_leaf():
            third_byte = 0
            forth_byte = tree.right.symbol
        
        else:
            third_byte = 1
            forth_byte = tree.right.number
        
        
    return get_byte(tree.left) + get_byte(tree.right) + [first_byte, second_byte, third_byte, forth_byte]


   

def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file to compress
    @param str out_file: output file to store compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in node_lst.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    
    def make_tree(root, tree):
        if root.l_type == 0: #Left is a leaf
            tree.left = HuffmanNode(root.l_data)
        if root.r_type == 0: #Right is a leaf
            tree.right = HuffmanNode(root.r_data)
        if root.l_type == 1: #Left is a parent
            tree.left = HuffmanNode()
            make_tree(node_lst[root.l_data], tree.left)
        if root.r_type == 1: #Right is a parent
            tree.right = HuffmanNode()
            make_tree(node_lst[root.r_data], tree.right)
        
    root = node_lst[root_index]
    node = HuffmanNode()        
    make_tree(root, node)
    return node


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that node_lst represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in 'node_lst'
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    # todo


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: number of bytes to decompress from text.
    @rtype: bytes
    """
    symbol_to_code = get_codes(tree) #Dictionary linking symbols to codes
    code_to_symbol = {}
    for key in symbol_to_code: #Creates dictionary linking codes to symbols
        code_to_symbol[symbol_to_code[key]] = key
    binary = ''
    original = []
    for byte in text:
        cur_bin = byte_to_bits(byte) #Binary rep of current byte
        binary += cur_bin
    x = 0
    for i in range(len(binary)):
        if binary[x:i] in code_to_symbol and size > 0: #Found binary in dictionary
            size -= 1
            original.append(code_to_symbol[binary[x:i]])
            x = i
    return bytes(original)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    #leafs = get_leaves(tree)
    
    temp = tree
    def swap_symbol(tree, sym1, sym2):
        '''
        this function takes in two symbols and a tree swaps the symbols
        of the two nodes with the given symbols
        '''
        if tree.symbol == sym1:
            tree.symbol = sym2
            
        elif tree.symbol == sym2:
            tree.symbol = sym1
            
        if tree.left != None:
            swap_symbol(tree.left, sym1, sym2)
            
        if tree.right != None:
            swap_symbol(tree.right, sym1, sym2)
    
    codes_dict = get_codes(tree)
    codes = []
    for symbol in codes_dict:
        codes.append((symbol, codes_dict[symbol]))

    for i in range(len(codes)-1):
        
        largest = i
        for j in range(i+1, len(codes)):
              if freq_dict[codes[j][0]] > freq_dict[codes[largest][0]]:
                  largest = j
                  
        if i != largest:
            temp = codes[i]
            codes[i] = codes[largest]
            codes[largest] = temp
            swap_symbol(tree, codes[i][0], codes[largest][0])
            



if __name__ == "__main__":
    # TODO: Uncomment these when you have implemented all the functions
    # import doctest
    # doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
