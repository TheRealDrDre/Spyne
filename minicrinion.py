# Mini crinion

from spyne.neural import Group, Projection, Circuit
import numpy as np
from spyne.learning import chl

# Questions

ALIVE = np.array([1, 0, 0, 0])
SMALL = np.array([0, 1, 0, 0])
DANGEROUS = np.array([0, 0, 1, 0])
AQUATIC = np.array([0, 0, 0, 1])

# Answers

YES = np.array([1, 0, 0, 0])
NO = np.array([0, 1, 0, 0 ])

# Words
#		        Alive	Small	Dangerous	Aquatic
#	Pirana	    Yes	    Yes	    Yes	        Yes
#	Scorpion	Yes	    Yes	    Yes	        No
#	Goldfish	Yes 	Yes	    No	        Yes
#	Ant	        Yes 	Yes	    No	        No
#	Shark   	Yes 	No	    Yes	        Yes
#	Bear    	Yes 	No	    Yes	        No
#	Dolphin	    Yes 	No	    No	        Yes
#	Sheep   	Yes 	No	    No	        No
#	Fishhook	No  	Yes	    Yes	        Yes
#	Knife   	No  	Yes	    Yes	        No
#	Snorkel 	No  	Yes	    No	        Yes
#	Yoyo	    No  	Yes	    No	        No
#	Torpedo 	No  	No	    Yes	        Yes
#	Bazooka 	No  	No	    Yes	        No
#	Boat	    No  	No	    No	        Yes
#	House	    No  	No	    No	        No

WORD_LIST = ("Pirana", "Scorpion", "Goldfish", "Ant", "Shark", "Bear",
             "Dolphin", "Sheep", "Fishhook", "Knife", "Snorkel", "Yoyo",
             "Torpedo", "Bazooka", "Boat", "House")

WORDS = { }  # Empty dictionary. Needs to be initialized by 'init_words'

def init_words():
    """
Initializes the list dictionary of words. Each words is associated to
a different random binary vector.
    """
    previously_used = []  # previously used patterns 
    for word in WORD_LIST:
        pattern = np.round(np.random.random((25, 1)))
        
        # Note that 'previously_used' contains the string
        # version of the array --- python gets otherwise confused
        # by checking the existence of an array inside an array...
        
        while np.array_str(pattern) in previously_used:
            pattern = np.round(np.random.random((25, 1)))
        
        WORDS[word] = pattern
        previously_used.append(np.array_str(pattern))

def words_initialized():
    if set(WORDS.keys()) == set(WORD_LIST):
        return True
    else:
        return False


INPUTS = []
OUTPUTS = []

def init_matrix():
    """Initializes the matrix of associatons between words and properties"""
    if words_initialized():
        index = 0
        for alive in [YES, NO]:
            for small in [YES, NO]:
                for dangerous in [YES, NO]:
                    for aquatic in [YES, NO]:
                        word = WORDS[ WORD_LIST[ index ] ]
                        INPUTS.append((word, ALIVE))
                        OUTPUTS.append(alive)
                        
                        INPUTS.append((word, SMALL))
                        OUTPUTS.append(small)
                        
                        INPUTS.append((word, DANGEROUS))
                        OUTPUTS.append(dangerous)
                    
                        INPUTS.append((word, AQUATIC))
                        OUTPUTS.append(aquatic)
                        
                        index += 1
        return True
    else:
        return False
                    

def create_model():
    """Creates a minicrinion model"""
    # No basal ganglia here
    
    # The groups
    word = Group(size=25, name="Word")
    response = Group(size=4, name="Response")
    semantic = Group(size=8, name="Semantic")
    question = Group(size=4, name="Question")
    
    # Prettifying for visual display
    word.geometry = (5, 5)
    response.geometry = (2, 2)
    semantic.geometry = (4, 2)
    question.geometry = (4, 1)

    
    # The Connections
    #
    # Word  Quest
    #  |   /  |
    #  +  +   |
    # Seman   |
    #     +   |
    #      \  |
    #       + +
    #       Resp
    
    w2s = word.ConnectTo(semantic)
    q2s = question.ConnectTo(semantic)
    s2r = semantic.ConnectTo(response)
    q2r = question.ConnectTo(response)
    r2s = response.ConnectTo(semantic)
    
    w2s.weights = np.random.random(w2s.weights.shape) / 10.0
    q2s.weights = np.random.random(q2s.weights.shape) / 10.0
    s2r.weights = np.random.random(s2r.weights.shape) / 10.0
    q2r.weights = np.random.random(q2r.weights.shape) / 10.0
    r2s.weights = np.random.random(r2s.weights.shape) / 10.0
    
    # Setting up the circuit
    
    C = Circuit()
    
    for g in [word, response, semantic, question]:
        C.AddGroup(g)
        g.SetContext(C)
        
    for p in [w2s, q2s, s2r, q2r, r2s]:
        p.SetContext(C)
        
    
    C.SetInput(word)
    C.SetInput(question)
    C.SetOutput(response)
    
    
    return C


                        
def train_model(model, max_epochs=10^4):
    """Trains a model"""
    chl(model, INPUTS[5], [OUTPUTS[5]], max_epochs=max_epochs)


if __name__ == "__main__":
    init_words()
    init_matrix()
    model = create_model()
    train_model(model, max_epochs=10)
    #return model
    