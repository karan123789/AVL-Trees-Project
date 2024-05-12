import math
from typing import TypeVar, Generator, List, Tuple, Optional
from collections import deque
import json
from queue import SimpleQueue
import heapq

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
# represents a Node object (forward-declare to use in Node __init__)
Node = TypeVar("Node")
# represents a custom type used in application
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")


class Node:
    """
    Implementation of an BST and AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return repr(self)


####################################################################################################

class BinarySearchTree:
    """
    Implementation of an BSTree.
    Modify only below indicated line.
    """

    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty BST tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty BST Tree"

        lines = pretty_print_binary_tree(self.origin, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __str__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    ########################################
    # Implement functions below this line. #
    ########################################

    def height(self, root: Node) -> int:
        """
        Calculates and returns the height of a subtree in the BSTree,
        handling the case where root is None. Note that an empty subtree
        has a height of -1.

        :param root: The root of the subtree whose height is to be calculated
        :return: The height of the subtree at root, or -1 if root is None.
        """
        if root is None:
          return -1
        else:
          return root.height

    def insert(self, root: Node, val: T) -> None:
        """
        Inserts a node with the value val into the subtree rooted at root,
        returning the root of the balanced subtree after the insertion.

        :param root: Root of the subtree where val is to be inserted
        :param val: The value to insert
        :return: None
        """
        if root is None:
            self.origin = Node(val)
            self.size += 1
            return

        if val == root.value:
            return root

        if val < root.value:
            if root.left is None:
                root.left = Node(val, parent=root)
                self.size += 1
            else:
                self.insert(root.left, val)
        elif val > root.value:
            if root.right is None:
                root.right = Node(val, parent=root)
                self.size += 1
            else:
                self.insert(root.right, val)
        else:
            return

        height_left = self.height(root.left)
        height_right = self.height(root.right)
        root.height = max(height_left, height_right) + 1

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Removes the node with the value val from the subtree rooted at root,
        and returns the root of the subtree after the removal.

        :param root: The root of the subtree from which to delete val
        :param val: The value to be deleted
        :return: The root of the new subtree after the removal (could be the original root).
        """
        if root is None:
            return root

        if val < root.value:
            root.left = self.remove(root.left, val)
            if root.left is not None:
                root.left.parent = root
        elif val > root.value:
            root.right = self.remove(root.right, val)
            if root.right is not None:
                root.right.parent = root
        else:
            if root.left is None:
                temp_root = root.right
                if temp_root:
                    temp_root.parent = root.parent
                if root == self.origin:
                    self.origin = temp_root
                self.size -= 1
                return temp_root
            elif root.right is None:
                temp_root = root.left
                if temp_root is not None:
                    temp_root.parent = root.parent
                if root == self.origin:
                    self.origin = temp_root
                self.size -= 1
                return temp_root

            pred_node = root.left
            while pred_node.right:
                pred_node = pred_node.right
            root.value = pred_node.value
            root.left = self.remove(root.left, pred_node.value)

        height_left = self.height(root.left)
        height_right = self.height(root.right)
        root.height = max(height_left, height_right) + 1

        return root




    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Searches for and returns the Node containing the value val in the subtree rooted at root.

        :param root: The root of the subtree in which to search for val
        :param val: The value to search for
        :return: Node containing val, or node below which val would be inserted if child does not exist
        """
        if root is None or root.value == val:
            return root


        if root.value < val:
            if root.right is None:
                return root
            return self.search(root.right, val)


        if root.left is None:
            return root
        return self.search(root.left, val)



class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string.

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        return super(AVLTree, self).__repr__()

    def __str__(self) -> str:
        """
        Represent the AVLTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    ########################################
    # Implement functions below this line. #
    ########################################

    def height(self, root: Node) -> int:
        """
        Calculates the height of a subtree in the AVL tree, handling cases
        where root might be None. Remember, the height of an empty subtree
        is defined as -1.

        :param root: Root node of the subtree whose height you wish to determine
        :return: Height of the subtree rooted at root
        """
        if root is None:
            return -1
        else:
            return root.height

    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        This method performs a left rotation on the subtree rooted at root
        ,returning the new root of the subtree after the rotation.

        :param root: Root node of the subtree that is to be rotated
        :return: Root of new subtree after rotation
        """
        if root is None or root.right is None:
            return root

        new_root = root.right
        root.right = new_root.left
        if new_root.left:
            new_root.left.parent = root

        new_root.parent = root.parent
        if root.parent is None:
            self.origin = new_root
        elif root == root.parent.left:
            root.parent.left = new_root
        else:
            root.parent.right = new_root

        new_root.left = root
        root.parent = new_root


        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))

        return new_root

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        This method performs a right rotation on the subtree rooted at root,
        returning the new root of the subtree after the rotation.

        :param root: Root node of the subtree that is to be rotated
        :return: Root node of the subtree after rotation
        """
        if root is None or root.left is None:
            return root

        new_root = root.left
        root.left = new_root.right
        if new_root.right:
            new_root.right.parent = root

        new_root.parent = root.parent
        if root.parent is None:
            self.origin = new_root
        elif root == root.parent.right:
            root.parent.right = new_root
        else:
            root.parent.left = new_root

        new_root.right = root
        root.parent = new_root


        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))

        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        This method computes the balance factor of the subtree rooted at root.
        :param root: Root node of the subtree on which to compute balance factor
        :return: Integer representing balance factor of root
        """
        if root is None:
            return 0
        return self.height(root.left) - self.height(root.right)

    def rebalance(self, root: Node) -> Optional[Node]:
        """
        Rebalances subtree rooted at root if unbalanced, returns root of resulting subtree post-rebalancing

        :param root: Root of the subtree that potentially needs rebalancing
        :return: Root of the new, potentially rebalanced subtree
        """
        balance = self.balance_factor(root)

        if balance >= 2:
            if self.balance_factor(root.left) < 0:
                root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        elif balance <= -2:
            if self.balance_factor(root.right) > 0:
                root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def insert(self, root: Node, val: T) -> Optional[Node]:
        """
        This function inserts a new node with value val into the subtree
        rooted at root, balancing the subtree as necessary, and returns
        the root of the resulting subtree.

        :param root: Root of the subtree where val is to be inserted
        :param val: Value to be inserted
        :return: Root of the new, balanced subtree
        """
        if root is None:
            self.origin = Node(val)
            self.size += 1
            return self.origin

        if val == root.value:
            return root

        if val < root.value:
            if root.left is None:
                root.left = Node(val, parent=root)
                self.size += 1
            else:
                self.insert(root.left, val)
        elif val > root.value:
            if root.right is None:
                root.right = Node(val, parent=root)
                self.size += 1
            else:
                self.insert(root.right, val)

        root.height = max(self.height(root.left), self.height(root.right)) + 1
        root = self.rebalance(root)

        return root

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        This function removes the node with value val from the subtree
        rooted at root, balances the subtree as necessary, and returns
        the root of the resulting subtree.

        :param root: Root of the subtree from which val is removed
        :param val: Value to be removed
        :return: Root of the new, balanced subtree
        """
        if root is None:
            return root

        if val < root.value:
            root.left = self.remove(root.left, val)
            if root.left is not None:
                root.left.parent = root
        elif val > root.value:
            root.right = self.remove(root.right, val)
            if root.right is not None:
                root.right.parent = root
        else:
            if root.left is None:
                temp_root = root.right
                if temp_root:
                    temp_root.parent = root.parent
                if root == self.origin:
                    self.origin = temp_root
                self.size -= 1
                return temp_root
            elif root.right is None:
                temp_root = root.left
                if temp_root is not None:
                    temp_root.parent = root.parent
                if root == self.origin:
                    self.origin = temp_root
                self.size -= 1
                return temp_root

            pred_node = root.left
            while pred_node.right:
                pred_node = pred_node.right
            root.value = pred_node.value
            root.left = self.remove(root.left, pred_node.value)

        height_left = self.height(root.left)
        height_right = self.height(root.right)
        root.height = max(height_left, height_right) + 1

        root = self.rebalance(root)
        return root

    def min(self, root: Node) -> Optional[Node]:
        """
        This function searches for and returns the Node 
        containing the smallest value within the subtree rooted at root.

        :param root: Root of the subtree within which to search for minimum value
        :return: Node containing the smallest in subtree rooted at root
        """
        if root is None or root.left is None:
            return root


        return self.min(root.left)

    def max(self, root: Node) -> Optional[Node]:
        """
        This function searches for and returns the Node containing the 
        largest value within the subtree rooted at root.

        :param root: Root of the subtree within which to search for minimum value
        :return: Node containing the largest in subtree rooted at root
        """
        if root is None or root.right is None:
            return root

        return self.max(root.right)

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        This function searches for the Node with the value val within the subtree rooted at root.

        :param root: Root of the subtree within which to search for val
        :param val: Value to be searched for within subtree rooted at root
        :return: Node containing val if it exists within subtree, else, Node would be inserted as a child
        """
        if root is None or root.value == val:
            return root

        if root.value < val:
            if root.right is None:
                return root
            return self.search(root.right, val)


        if root.left is None:
            return root
        return self.search(root.left, val)

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs an inorder traversal (left, current, right) 
        of the subtree rooted at root, generating the nodes one at a time using a Python generator

        :param root: Root node of the subtree being traversed
        :return: Generator yielding the nodes of subtree in inorder
        """
        if root:
            yield from self.inorder(root.left)
            yield root
            yield from self.inorder(root.right)

    def __iter__(self) -> Generator[Node, None, None]:
        """
        Makes the AVL tree class iterable

        :return: Generator yielding nodes of the tree in inorder
        """
        return self.inorder(self.origin) 

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a preorder traversal (current, left, right) of the subtree rooted at root, 
        generating the nodes one at a time using a Python generator.

        :param root: Root node of the subtree being traversed
        :return: Generator yielding the nodes of subtree in preorder
        """
        if root is None:
          return
        yield root
        yield from self.preorder(root.left)
        yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a postorder traversal (left, right, current) 
        of the subtree rooted at root, generating the nodes one at a time using a Python generator.

        :param root: Root node of the subtree being traversed
        :return: Generator yielding the nodes of subtree in postorder
        """
        if root is None:
          return
        yield from self.postorder(root.left)
        yield from self.postorder(root.right)
        yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a level-order (breadth-first) traversal of the subtree rooted at root, 
        generating the nodes one at a time using a Python generator.

        :param root: Root node of the subtree being traversed
        :return: Generator yielding the nodes of subtree in levelorder
        """
        if root is None:
            return
        
        queue = SimpleQueue()
        queue.put(root)

        while not queue.empty():
            node = queue.get()
            yield node

            if node.left is not None:
                queue.put(node.left)
            if node.right is not None:
                queue.put(node.right)


####################################################################################################

class User:
    """
    Class representing a user of the stock marker.
    Note: A user can be both a buyer and seller.
    """

    def __init__(self, name, pe_ratio_threshold, div_yield_threshold):
        self.name = name
        self.pe_ratio_threshold = pe_ratio_threshold
        self.div_yield_threshold = div_yield_threshold


####################################################################################################

class Stock:
    __slots__ = ['ticker', 'name', 'price', 'pe', 'mkt_cap', 'div_yield']
    TOLERANCE = 0.001

    def __init__(self, ticker, name, price, pe, mkt_cap, div_yield):
        """
        Initialize a stock.

        :param name: Name of the stock.
        :param price: Selling price of stock.
        :param pe: Price to earnings ratio of the stock.
        :param mkt_cap: Market capacity.
        :param div_yield: Dividend yield for the stock.
        """
        self.ticker = ticker
        self.name = name
        self.price = price
        self.pe = pe
        self.mkt_cap = mkt_cap
        self.div_yield = div_yield

    def __repr__(self):
        """
        Return string representation of the stock.

        :return: String representation of the stock.
        """
        return f"{self.ticker}: PE: {self.pe}"

    def __str__(self):
        """
        Return string representation of the stock.

        :return: String representation of the stock.
        """
        return repr(self)

    def __lt__(self, other):
        """
        Check if the stock is less than the other stock.

        :param other: The other stock to compare to.
        :return: True if the stock is less than the other stock, False otherwise.
        """
        return self.pe < other.pe

    def __eq__(self, other):
        """
        Check if the stock is equal to the other stock.

        :param other: The other stock to compare to.
        :return: True if the stock is equal to the other stock, False otherwise.
        """
        return abs(self.pe - other.pe) < self.TOLERANCE


def make_stock_from_dictionary(stock_dictionary: dict[str: str]) -> Stock:
    """
    Builds an AVL tree with the given stock dictionary.

    :param stock_dictionary: Dictionary of stocks to be inserted into the AVL tree.
    :return: A stock in a Stock object.
    """
    stock = Stock(stock_dictionary['ticker'], stock_dictionary['name'], stock_dictionary['price'], \
                  stock_dictionary['pe_ratio'], stock_dictionary['market_cap'], stock_dictionary['div_yield'])
    return stock

def build_tree_with_stocks(stocks_list: List[dict[str: str]]) -> AVLTree:
    """
    Builds an AVL tree with the given list of stocks.

    :param stocks_list: List of stocks to be inserted into the AVL tree.
    :return: AVL tree with the given stocks.
    """
    avl = AVLTree()
    for stock in stocks_list:
        stock = make_stock_from_dictionary(stock)
        avl.insert(avl.origin, stock)
    return avl


####################################################################################################
# Implement functions below this line. #
####################################################################################################

def recommend_stock(stock_tree: AVLTree, user: User, action: str) -> Optional[Stock]:
    """
    This function analyzes stocks within an AVL Tree to identify the most suitable stock to buy or sell,
    according to the user’s predefined criteria such as the P/E ratio and dividend yield.

    :param stock_tree: AVL Tree containing stock nodes
    :param user: user object representing the investor's preferences
    :param action: String indicating the desired action, either 'buy' or 'sell'
    :return: The function returns a Stock object representing the recommended stock that best fits
    the user’s criteria. If no stock meets the criteria, None is returned.
    """
    best = [None]

    def traverse_list(node: Node):
        if node is None:
            return None
        if action == 'buy':
            traverse_list(node.left)
            if (node.value.pe <= user.pe_ratio_threshold and node.value.div_yield >= user.div_yield_threshold) \
                    and (best[0] is None or node.value.pe < best[0].pe):
                best[0] = node.value
            traverse_list(node.right)
        elif action == 'sell':
            traverse_list(node.right)
            if (node.value.pe > user.pe_ratio_threshold or node.value.div_yield < user.div_yield_threshold) \
                    and (best[0] is None or node.value.pe > best[0].pe):
                best[0] = node.value
            traverse_list(node.left)

    traverse_list(stock_tree.origin)
    return best[0]



def prune(stock_tree: AVLTree, threshold: float = 0.05) -> None:
    """
    This function removes subtrees of the given Stock AVL Tree where 
    all pe values are less than threshold.

    :param stock_tree: AVL Tree to be pruned
    :param threshold: Threshold for pruning
    :return: None
    """
    def prune_recursive(node: Node):
        if node is None:
            return None

        prune_recursive(node.left)
        prune_recursive(node.right)

        if node is not None and node.value.pe < threshold:
            temp_node = node.parent

            if temp_node is not None:
                if temp_node.left == node:
                    temp_node.left = stock_tree.remove(temp_node.left, node.value)
                else:
                    temp_node.right = stock_tree.remove(temp_node.right, node.value)
            else:
                stock_tree.remove(node, node.value)

            return

        if node is not None:
            stock_tree.rebalance(node)

    prune_recursive(stock_tree.origin)




####################################################################################################
####################### EXTRA CREDIT ##############################################################
####################################################################################################

class Blackbox:
    def __init__(self):
        """
        Initialize a minheap.
        """
        self.heap = []

    def store(self, value: T):
        """
        Push a value into the heap while maintaining minheap property.

        :param value: The value to be added.
        """
        heapq.heappush(self.heap, value)

    def get_next(self) -> T:
        """
        Pop minimum from min heap.

        :return: Smallest value in heap.
        """
        return heapq.heappop(self.heap)

    def __len__(self):
        """
        Length of the heap.

        :return: The length of the heap
        """
        return len(self.heap)

    def __repr__(self) -> str:
        """
        The string representation of the heap.

        :return: The string representation of the heap.
        """
        return repr(self.heap)

    __str__ = __repr__


class HuffmanNode:
    __slots__ = ['character', 'frequency', 'left', 'right', 'parent']

    def __init__(self, character, frequency):
        self.character = character
        self.frequency = frequency

        self.left = None
        self.right = None
        self.parent = None

    def __lt__(self, other):
        """
        Checks if node is less than other.

        :param other: The other node to compare to.
        """
        return self.frequency < other.frequency

    def __repr__(self):
        """
        Returns string representation.

        :return: The string representation.
        """
        return '<Char: {}, Freq: {}>'.format(self.character, self.frequency)

    __str__ = __repr__


class HuffmanTree:
    __slots__ = ['root', 'blackbox']

    def __init__(self):
        self.root = None
        self.blackbox = Blackbox()

    def __repr__(self):
        """
        Returns the string representation.

        :return: The string representation.
        """
        if self.root is None:
            return "Empty Tree"

        lines = pretty_print_binary_tree(self.root, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    __str__ = __repr__

    def make_char_map(self) -> dict[str: str]:
        """
        Create a binary mapping from the huffman tree.

        :return: Dictionary mapping from characters to "binary" strings.
        """
        mapping = {}

        def traversal(root: HuffmanNode, current_str: str):
            if not root:
                return

            if not root.left and not root.right:
                mapping[root.character] = current_str
                return

            if root.left:
                traversal(root.left, current_str=current_str + '0')

            if root.right:
                traversal(root.right, current_str=current_str + '1')

        traversal(self.root, '')

        return mapping

    def compress(self, input: str) -> tuple[dict[str: str], List[str]]:
        """
        Compress the input data by creating a map via huffman tree.

        :param input: String to compress.
        :return: First value to return is the mapping from characters to binary strings.
        Second value is the compressed data.
        """
        self.build(input)

        mapping = self.make_char_map()

        compressed_data = []

        for char in input:
            compressed_data.append(mapping[char])

        return mapping, compressed_data

    def decompress(self, mapping: dict[str: str], compressed: List[str]) -> str:
        """
        Use the mapping from characters to binary strings to decompress the array of bits.

        :param mapping: Mapping of characters to binary strings.
        :param compressed: Array of binary strings that are encoded.
        """

        reverse_mapping = {v: k for k, v in mapping.items()}

        decompressed = ""

        for encoded in compressed:
            decompressed += reverse_mapping[encoded]

        return decompressed

    ########################################################################################
    # Implement functions below this line. #
    ########################################################################################

    def build(self, chars: str) -> None:
        """
        Given some input, construct a Huffman tree based off that input

        :param chars: string to create a Huffman tree based off of
        :return: None
        """

        freq_table = {}
        for char in chars:
          if char in freq_table:
            freq_table[char] += 1
          else:
            freq_table[char] = 1

        for char, freq in freq_table.items():
          node = HuffmanNode(char, freq)
          self.blackbox.store(node)

        while len(self.blackbox) > 1:
          node1 = self.blackbox.get_next()
          node2 = self.blackbox.get_next()

          merged = HuffmanNode(None, node1.frequency + node2.frequency)
          merged.left = node1
          merged.right = node2

          self.blackbox.store(merged)

        self.root = self.blackbox.get_next()


def pretty_print_binary_tree(root: Node, curr_index: int, include_index: bool = False,
                             delimiter: str = "-", ) -> \
        Tuple[List[str], int, int, int]:
    """
    Taken from: https://github.com/joowani/binarytree

    Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    :param root: Root node of the binary tree.
    :type root: binarytree.Node | None
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param include_index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type include_index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if include_index:
        node_repr = "{}{}{}".format(curr_index, delimiter, root.value)
    else:
        if type(root) == HuffmanNode:
            node_repr = repr(root)
        elif type(root.value) == AVLWrappedDictionary:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value.key) if root.parent else "None"}'
        else:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value) if root.parent else "None"}'

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = pretty_print_binary_tree(
        root.left, 2 * curr_index + 1, include_index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = pretty_print_binary_tree(
        root.right, 2 * curr_index + 2, include_index, delimiter
    )

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


if __name__ == "__main__":
    pass
