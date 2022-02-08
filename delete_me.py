class TeamIterator:
   ''' Iterator class '''
   def __init__(self, team):
       # Team object reference
       self._team = team
       # member variable to keep track of current index
       self._index = 0
   def __next__(self):
       ''''Returns the next value from team object's lists '''
       if self._index < (len(self._team._juniorMembers) + len(self._team._seniorMembers)) :
           if self._index < len(self._team._juniorMembers): # Check if junior members are fully iterated or not
               result = (self._team._juniorMembers[self._index] , 'junior')
           else:
               result = (self._team._seniorMembers[self._index - len(self._team._juniorMembers)]   , 'senior')
           self._index +=1
           return result
       # End of Iteration
       raise StopIteration
class Team:
   '''
   Contains List of Junior and senior team members and also overrides the __iter__() function.
   '''
   def __init__(self):
       self._juniorMembers = list()
       self._seniorMembers = list()
   def addJuniorMembers(self, members):
       self._juniorMembers += members
   def addSeniorMembers(self, members):
       self._seniorMembers += members
   def __iter__(self):
       ''' Returns the Iterator object '''
       return TeamIterator(self)

class NumbersIterator:
    
    def __init__(self, numbers):
        
        self._numbers = numbers
        self._index = 0
    
        self._n_pos_nums = len(self._numbers._positive_numbers)
        self._n_neg_nums = len(self._numbers._negative_numbers)
    
    def __next__(self):
        
        if self._index < self._n_pos_nums:
            
            num = self._numbers._positive_numbers[self._index] ** 2
            self._index += 1
            return num
        
        elif self._index < self._n_pos_nums + self._n_neg_nums:
            
            num = self._numbers._negative_numbers[self._index - self._n_pos_nums] ** 2
            self._index += 1
            return num
        
        else:
            
            raise StopIteration


class Numbers:
    
    def __init__(self):
        self._positive_numbers = list()
        self._negative_numbers = list()
    
    def add_numbers(self, numbers):
        
        for num in numbers:
            if num > 0:
                self._positive_numbers.append(num)
            else:
                self._negative_numbers.append(num)
    
    def __iter__(self):
        return NumbersIterator(self)

numbers = Numbers()

numbers.add_numbers(range(-10,11))

for num in numbers:
    print(num)