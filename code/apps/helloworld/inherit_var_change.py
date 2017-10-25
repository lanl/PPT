'''
Author:     @Nath
Date:       23 May 2017
Purpose:    Testing program to modify the processors with python inheritence.
'''


class Parent(object):
    '''
    To all PPT folks except @Nath, please ignore this file.
    This is Nath's toy for testing.
    '''
    def __init__(self):
        self.foo = ['Hello']
        self.name = 'Gopinath'
        self.newname = 'Nath'
    def print_vars(self):
        print "Variables : ", "foo = ",self.foo, "name = ",self.name, "newname = ",self.newname

class Child(Parent):
    def __init__(self):
        super(Child, self).__init__()
        self.foo.append('World')
        self.name = 'Robin'
        self.newname = 'Robert'
    def print_vars2(self):
        print "Child Variables : ", "foo = ",self.foo, "name = ",self.name, "newname = ",self.newname

class Child2(Child):
    def __init__(self):
        super(Child2, self).__init__()
        self.foo.append('No World')
        self.name = 'Child of Robin'
        self.newname = 'Child of Robert'
    def print_vars3(self):
        print "Child2 Variables : ", "foo = ",self.foo, "name = ",self.name, "newname = ",self.newname

pobj = Parent()
pobj.print_vars()
obj = Child()
#print "Child Variables : ", obj.foo, obj.name, obj.newname
obj.print_vars()
obj.print_vars2()
obj2 = Child2()
#obj2.print_vars2()
obj2.print_vars3()
