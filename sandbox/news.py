
class News:
    def __init__(self, text, stance):
        self.text = text
        assert stance in set(["positive", "negative", "neutral"]), "Invalid stance"
        self.stance = stance
    
    def get_str(self):
        return self.text
    
    def get_stance(self):
        return self.stance
    

        
