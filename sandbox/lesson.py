class Lesson():
    def __init__(self, lesson_text, lesson_time, lesson_importance, time_decay_rate=0.9):
        self.text = lesson_text
        self.time = lesson_time
        self.importance = lesson_importance
        self.time_decay_rate = time_decay_rate
    
    def score(self, current_time):
        return self.importance + self.time_decay_rate ** (current_time - self.time)