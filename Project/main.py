#import packages
import mysql.connector as mdb
import sys
from datetime import datetime
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.config import Config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# define varibale that will help us to run the code
host_k=None
user_k=None
password_k=None
num_in=None
openstore=None
closestore=None
camera=None

# load our serialized face detector model from disk
prototxtPath = './face_detector/deploy.prototxt'
weightsPath = './face_detector/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")


class Openscreen(BoxLayout):
    '''
    Openscreen is the home page of the app
    '''
    # Variable from KV file
    customer_k=ObjectProperty
    def __init__(self, **kwargs):
        '''
        Builder of the class
        :param kwargs:
        '''
        # Call builder of BoxLayout because we don't use default builder
        super(Openscreen, self).__init__(**kwargs)
        # Using global variable to check if the worker connected to the database -
        # If he did - the customer button won't be disabled
        global host_k
        if host_k!=None:
            self.customer_k.disabled=False
    def worker(self):
        '''
        # the worker will sent to Connect screen every time he press the worker button
        :return:
        '''
        self.clear_widgets()  # Clears Openscreen
        self.add_widget(Connect())  # Adds Connect as new widget


    def customer(self):
        '''
        Sending customer to Choose screen
        :return:
        '''
        self.clear_widgets()  # Clears Openscreen
        self.add_widget(Choose())  # Adds Choose as new widget

    def exitout(self):
        'Close application'
        sys.exit()

class Connect(BoxLayout):
    '''
    Connect screen - the worker will connect to database
    while entering the correct host, user and password
    '''
    # Variables from KV file
    host = ObjectProperty()
    user = ObjectProperty()
    password = ObjectProperty()

    def work_connect(self):
        '''
        Creating and connecting to the database and creating the table - if not exist
        :return:
        '''
        # Using try and except to prevent the system from crashing
        try:
            global host_k
            host_k = self.host.text
            global user_k
            user_k = self.user.text
            global password_k
            password_k = self.password.text
            # Creates database and table if not exists yet
            # connecting to DB with worker host, username and password
            con = mdb.connect(host=str(self.host.text), user=str(self.user.text), passwd=str(self.password.text))
            cur = con.cursor() # Making cursor
            # make the DB if not exist and it name is checkIt
            cur.execute('CREATE DATABASE if not exists maskup;')
            con.commit()  # Makes transaction
            con.close()  # Close transaction
            # Connecting to DB with worker host, username and password
            con = mdb.connect(host=str(self.host.text), user=str(self.user.text), passwd=str(self.password.text),db='maskup')
            cur = con.cursor() # Making cursor
            # make the table in the DB if not exist and it name is customer.
            # the col are: customer_name, customer_first_name, customer_last_name, customer_phone,
            # in_date, in_hour, out_hour
            cur.execute("CREATE TABLE IF NOT EXISTS customer \
                                        (customer_number INTEGER(255) NOT NULL PRIMARY KEY AUTO_INCREMENT, customer_first_name varchar(10) NOT NULL, customer_last_name varchar(10) NOT NULL, customer_phone varchar(10) NOT NULL , in_date varchar(20) NOT NULL ,  in_hour varchar(20) NOT NULL , out_hour varchar(20)) ")
            con.commit()  # Makes transaction
            con.close()  # Close transaction

            self.clear_widgets()  # Clears Connect
            self.add_widget(Control())  # Adds Control as new widget
        except: # if the worker entered wrong details to connect to database - error mas for incorrect input
            self.host.text=self.user.text=self.password.text="Please try again"


    def home(self):
        self.clear_widgets()  # Clears Connect
        self.add_widget(Openscreen())  # Adds Openscreen as new widget

class TextInputPopup(Popup):
    '''
    Popup screen that will show title and label
    '''
    # Variables from KV file
    title = StringProperty()
    label = StringProperty()

    def __init__(self, title, label, **kwargs):
        '''
        Builder of the class
        :param title: the title of the Popup, that will show on top of the screen
        :param label: the label that will be displayed in the Popup screen
        :param kwargs:
        '''
        # Call builder of BoxLayout because we don't use default builder
        super(TextInputPopup, self).__init__(**kwargs)
        # Triggers the set_description function
        self.set_description(title, label)

    def set_description(self, title, label):
        '''
        set the description of the Popup screen
        :param title: the title of the Popup, that will show on top of the screen
        :param label: the label that will be displayed in the Popup screen
        :return:
        '''
        self.title = title
        self.label = label

class Control(BoxLayout):
    '''
    Insert the limit number of customers that can be in the store at the same time
    checking by date and time which customers were in the store at that time
    checking how many customers are currently in the store
    '''
    # Variables from KV file
    day= ObjectProperty()
    month= ObjectProperty()
    year= ObjectProperty()
    start_hour= ObjectProperty()
    start_minute= ObjectProperty()
    finish_hour= ObjectProperty()
    finish_minute= ObjectProperty()
    limit = ObjectProperty()


    def __init__(self):
        '''
        Builder of the class
        '''
        # Call builder of BoxLayout because we don't use default builder
        super(Control,self).__init__()
        # Using global variables to connect to the DB
        global host_k
        self.host=host_k
        global user_k
        self.user=user_k
        global password_k
        self.password=password_k
        # Limits the length of the Spinner menu that be displayed
        self.day.dropdown_cls.max_height = 200


    def sickdate(self):
        '''
        Insert the data that the worker entered
        into new variables that will sent to Showsick
        :return:
        '''
        # define variables (date, start_time and finish_time)
        # that will help to find which customer has been in the store
        # at the same time
        date=self.year.text +'-'+self.month.text+'-'+self.day.text
        start_time=self.start_hour.text +':'+self.start_minute.text+':'+'0'
        finish_time=self.finish_hour.text +':'+self.finish_minute.text+':'+'0'
        self.clear_widgets()  # Clears Control
        self.add_widget(Showsick(date,start_time,finish_time,self.host,self.user,self.password))  # Adds Showsick as new form


    def limit_num(self):
        '''
        Insert the limit number of customers that can be at the store at the same time
        to new DB and table
        :return:
        '''
        # Connecting to DB with worker host, username and password
        con = mdb.connect(host=self.host, user=self.user, passwd=self.password,db='maskup')
        cur = con.cursor() # Making cursor
        # make the table in the DB if not exist and it name is cuslimit. the col is customer_limit
        cur.execute("CREATE TABLE IF NOT EXISTS cuslimit \
                                                (customer_limit INTEGER(100) NOT NULL  )")
        con.commit()  # Makes transaction
        con.close()  # Close transaction
        # Connecting to DB with worker host, username and password
        con = mdb.connect(host=self.host, user=self.user, passwd=self.password,db='maskup')
        cur = con.cursor() # Making cursor
        if self.limit.text.isdigit() :  # Check input
            cur.execute("INSERT INTO cuslimit (customer_limit) VALUES (%d)"
                        % (int(self.limit.text)))  # Inserts worker input to database
            con.commit() # Makes transaction
            con.close() # Close transaction
        else:
            self.limit.text= 'Please type valid data. Use 0-9 digits'  # Error msg for incorrect input.
            return
        self.limit.text = ''  # Clean text boxes
        self.limit.focus = True  # Set focus on limit customer text box for another insertion.


    def customer_in(self):
        '''
        Find how many customers are currently in the store
        :return:
        '''
        # Connecting to DB with worker host, username and password
        con = mdb.connect(host=self.host, user=self.user, passwd=self.password,db='maskup')
        cur = con.cursor() # Making cursor
        cur.execute(
            "SELECT count(customer_phone)FROM customer WHERE out_hour IS NULL  ") # Quering databse - present how many customers are currently in the store
        # Using global variable that contain how many customers are currently in the stor
        global num_in
        num_in= (cur.fetchone())[0]

    def on_call_popup(self):
        '''
        Show how many customers are currently in the store in Popup screen
        :return:
        '''
        # Trigger the customer_in function
        self.customer_in()
        # Using global variable
        global num_in
        # Define variable , using TextInputPopup class,
        # that makes Popup screen with will show
        # how many customers are currently in the store
        num_customer_now = TextInputPopup('The number customers in store', 'There are ' +str(num_in ) + ' customers in the store',auto_dismiss=False)
        # Openning the Popup screen
        num_customer_now.open()


    def back(self):
        '''
        Returns to the previous screen
        :return:
        '''
        self.clear_widgets()  # Clears Control
        self.add_widget(Connect()) # Adds Connect as new widget


    def home(self):
        '''
        Returns to the home screen
        :return:
        '''
        self.clear_widgets()  # Clears Control
        self.add_widget(Openscreen())  # Adds Openscreen as new form







class Showsick(BoxLayout):
    '''
    Show which customers have been at the store - content of DB
    '''
    # Variables from KV file
    res=ObjectProperty()
    def __init__(self,date,start_time,finish_time,host,user,password):
        '''
        Builder of the class
        :param date: the date that the worker insert
        :param start_time: the start time that the worker insert
        :param finish_time: the finish time that the worker insert
        :param host: the host that help to connect to DB that the worker insert
        :param user: the username that help to connect to DB that the worker insert
        :param password: the password that help to connect to DB that the worker insert
        '''
        super(Showsick, self).__init__()  # Call builder of BoxLayout because we don't use default builder
        self.date=date
        self.start_time=start_time
        self.finish_time=finish_time
        self.host=host
        self.user=user
        self.password=password
        # Connecting to DB with worker host, username and password
        con = mdb.connect(host=self.host, user=self.user, passwd=self.password, db='maskup')
        cur = con.cursor()  # Making cursor
        cur.execute(
            "SELECT customer_phone,customer_first_name,customer_last_name FROM customer WHERE in_date='%s' AND out_hour>'%s' AND in_hour<'%s'"
            % (date, start_time, finish_time)) # Quering DB - present the phone number and customer name
        rows= cur.fetchall() # Retrive all rows from query
        # Call method to show resulte
        self.show(rows)

    def show(self,rows):
        '''
        Shows results on screen
        :param rows: the rows that will present on the screen
        :return:
        '''
        # Clean previous results
        self.res.text=''
        # If there is no results to show
        if not rows:
            self.res.text='No records' # Error msg
        else:
            for r in rows:
                # Make readable msg
                self.res.text+=r[1]+' '+r[2]+' '+'0'+str(r[0])+'\n'

    def back(self):
        '''
        Returns to the previous screen
        :return:
        '''
        self.clear_widgets()  # Clears Showsick
        self.add_widget(Control())  # Adds Control as new widget

    def home(self):
        '''
        Returns to the home screen
        :return:
        '''
        self.clear_widgets()  # Clears Showsick
        self.add_widget(Openscreen())  # Adds Openscreen as new widget

class Choose(BoxLayout):
    '''
    Customer choose if he enters the store or exits
    '''

    def __init__(self):
        '''
        Builder of the class
        '''
        # Global variables that help connect to DB
        global host_k
        global user_k
        global password_k
        # Variables from KV file
        self.enter = ObjectProperty
        # Call builder of BoxLayout because we don't use default builder
        super(Choose,self).__init__()
        # Creates database and table if not exists yet
        # connecting to DB with worker username and password
        con = mdb.connect(host=host_k, user=user_k, passwd=password_k)
        cur = con.cursor() # Making cursor
        # make the DB if not exist and it name is maskup
        cur.execute('CREATE DATABASE if not exists maskup;')
        con.commit()  # Makes transaction
        con.close()  # Close transaction
        # Connecting to DB with worker host, username and password
        con = mdb.connect(host=host_k, user=user_k, passwd=password_k, db='maskup')
        cur = con.cursor() # Making cursor
        # make the table in the DB if not exist and it name is customer.
        # the col are: customer_name, customer_first_name, customer_last_name, customer_phone,
        # in_date, in_hour, out_hour
        cur.execute("CREATE TABLE IF NOT EXISTS customer \
                            (customer_number INTEGER(255) NOT NULL PRIMARY KEY AUTO_INCREMENT, customer_first_name varchar(10) NOT NULL, customer_last_name varchar(10) NOT NULL, customer_phone varchar(10) NOT NULL , in_date varchar(20) NOT NULL ,  in_hour varchar(20) NOT NULL , out_hour varchar(20)) ")
        con.commit()  # Makes transaction
        con.close()  # Close transaction
        # Connecting to DB with worker host, username and password
        con = mdb.connect(host=host_k, user=user_k, passwd=password_k, db='maskup')
        cur = con.cursor()  # Making cursor
        cur.execute(
            "SELECT count(customer_phone)FROM customer WHERE out_hour IS NULL  ")  # Quering databse - present how many customers are currently in the store
        num_customer = (cur.fetchone()) # Retrive single row from query
        # Using try and except to prevent the system from crashing
        try:
            # Connecting to DB with worker host, username and password
            con = mdb.connect(host=host_k, user=user_k, passwd=password_k, db='maskup')
            cur = con.cursor()  # Making cursor
            cur.execute(
                "SELECT(customer_limit)FROM cuslimit") # Quering DB - present the customer limit that the worker insert
            num_limit = ((cur.fetchall())[-1])[0] # Retrive the last row from query - the first one in the tuple
        except:
            num_limit=10 # If the worker didn't write customer limit - the default is 10
        # Checking if there are customers in the store - if so,
        # checking if there are more or less than the limit
        # if there is less - the next customer can press the enter button
        # else - the customer can't press the button and can't enter
        if num_customer==None or num_customer[0]<num_limit :
            self.enter.disabled = False


    def enterbutton(self):
        '''
        Sending to the next screen that continue the process of entering the store
        :return:
        '''
        self.clear_widgets()  # Clears Choose
        # Using try and except to prevent the system from crashing
        try:
            # Variables that trigger the start_work function in Cameramask class
            # and contain the result of the function
            result = Cameramask.start_work(self)
            # If the result from the function is good
            # sending to the next screen - InputForm
            if result=='good':
                self.add_widget(InputForm()) # Adds InputForm as new widget
            # If the result is bad - sending to screen with error msg
            else:
                self.add_widget(Nomask()) # Adds Nomask as new widget
        except:
            self.add_widget(Nomask()) # Adds Nomask as new widget

    def exitbutton(self):
        '''
        Sending to the next screen that continue the process of exiting the store
        :return:
        '''
        self.clear_widgets()  # Clears Choose
        self.add_widget(OutputForm())  # Adds OutputForm as new form


    def home(self):
        '''
        Returns to the home screen
        :return:
        '''
        self.clear_widgets()  # Clears Choose
        self.add_widget(Openscreen())  # Adds Openscreen as new widget


class Cameramask():
    '''
    Opens the camera and check if the customer wear mask on his face
    '''
    def start_work(self):
        '''
        Opens the camers and uses another function to check if the customer wear mask
        :return:
        '''
        # Creating new object of camera
        vs = cv2.VideoCapture(0)
        # Variables that will help to close the camera after X seconds
        lst_check = []
        lst_no_face=[]
        label = None
        # Loop over the frames from the video stream
        while True:
            result = None
            # Grab the frame from the threaded video stream
            frame = vs.read()[1]
            # Detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = Cameramask.detect_and_predict_mask(self, frame, faceNet, maskNet)
            # Loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # Unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                # Determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                # Include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                # Appending the label to list - it will help up tp close the camera after X seconds
                lst_check.append(label)
                # Display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Display the resulting frame
            cv2.imshow("Mask up!", frame)
            # Wait for press
            key = cv2.waitKey(1) & 0xFF

            # If the camera doesn't recognize face -
            # appending the label to no face list
            if label==None:
                lst_no_face.append(label)
            # If the camera didn't find face over 40 frames - the result is bad
            # and break from the loop
            if len(lst_no_face)>40:
                result='bad'
                break
            # if the camera recognize face
            # and recognize the customer wear mask correctly (on his mouth and nose)
            # and there is over 99% that the costumer wear mask
            # the result is good and
            # break from the loop
            if label!=None:
                if str(label[:4]) == 'Mask':
                    if str(label[6:-1]) > '99.00':
                        result = 'good'
                        print('in mask', result)
                        break
            # If there have been more than 40 frames that the customer didn't wear mask
            # the result is bad and
            # break from the loop
            if len(lst_check) > 40:
                result = 'bad'
                break
        # Release memory
        vs.release()
        cv2.destroyAllWindows()
        return result

    def detect_and_predict_mask(self, frame, faceNet, maskNet):
        '''
        Helps finding the customer face and checks if he wears mask
        :param frame: frame from the threaded video stream
        :param faceNet: face detector model
        :param maskNet:face mask detector model
        :return:
        '''
        # Grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        # Pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        # Initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
            # Filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # Compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # Ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                # Extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                # Add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        # Only make a predictions if at least one face was detected
        if len(faces) > 0:
            # For faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            preds = maskNet.predict(faces)
        # Return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)


class Nomask(BoxLayout):
    '''
    Shows error msg on screen - that the customer doesn't have mask face
    '''
    def __init__(self):
        '''
        Builder of the class
        '''
        # Call builder of BoxLayout because we don't use default builder
        super(Nomask,self).__init__()

    def home(self):
        '''
        Returns to the home screen
        :return:
        '''
        self.clear_widgets()  # Clears Nomask
        self.add_widget(Openscreen())  # Adds Openscreen as new widget


class InputForm(BoxLayout):
    '''
    Input form customer entering form
    '''
    first_name = ObjectProperty() # Variables from KV file.
    last_name = ObjectProperty() # Variables from KV file.
    phone=ObjectProperty() # Variables from KV file.

    def insert(self):
        '''
        Inserts data to database
        :return:
        '''
        # Using global variables to connect to the DB
        global host_k
        global user_k
        global password_k
        # Connecting to DB with worker host, username and password
        con = mdb.connect(host=host_k, user=user_k, passwd=password_k, db='maskup')
        cur = con.cursor()  # Making cursor
        if self.first_name.text.isalpha() and self.last_name.text.isalpha() and self.phone.text.isdigit() and len(self.phone.text)==10 and self.phone.text[0:2]=='05':  # Check input
            cur.execute("INSERT INTO customer (customer_first_name,customer_last_name,customer_phone,in_date,in_hour) VALUES ('%s','%s',%s, %s,%s)"
                        % (self.first_name.text,self.last_name.text,self.phone.text,("'"+(str(datetime.now())[0:11])+"'" ),("'"+(str(datetime.now())[11:])+"'" ))) # Inserts user input to database
            con.commit() # Makes transaction
            con.close() # Close transaction
        else:
            self.first_name.text = self.last_name.text = 'Please type valid data. Use a-z characters'  # Error msg for incorrect input.
            self.phone.text ='Please type valid phone number. Use 0-9 digits and write full number (10 digits) ' # Error msg for incorrect input.
            return
        self.first_name.text = self.last_name.text = self.phone.text = ''  # Clean text boxes
        self.first_name.focus = True  # Set focus on first name text box for another insertion.
        self.clear_widgets()  # Clears InputForm
        self.on_call_popup_storeopen() # Trigger on_call_popupa_storeopen - Popup screen
        self.add_widget(Openscreen()) # Adds Openscreen as new widget

    def on_call_popup_storeopen(self):
        '''
        Shows the Popup screen
        :return:
        '''
        global openstore
        # Creating Popup object that will show how the store's door opens
        openstore = Popup(title='     ', content=Storeopen(),background='michael.jpeg')
        # Open the Popup screen and show it to the customer
        openstore.open()

    def home(self):
        '''
        Returns to the home screen
        :return:
        '''
        self.clear_widgets()  # Clears InputForm
        self.add_widget(Openscreen())  # Adds Openscreen as new form


class Door(Widget):
    '''
    Store door
    '''
    pass


class Bar(Widget):
    '''
    Store bar
    '''
    pass


class Store(Widget):
    '''
    Store car
    '''
    pass


class Limit(Widget):
    '''
    Door limit switch.
    '''
    pass


class DoorBar(Widget):
    '''
    Door limit for door open, door close.
    '''
    pass

class Storeopen(Widget):
    '''
    Shows the store's door open
    '''
    # Variables from KV file.
    body = ObjectProperty
    delet = ObjectProperty
    doorBar = ObjectProperty
    doorOpen = ObjectProperty
    doorClose = ObjectProperty

    def __init__(self,**kwargs):
        '''
        Builder of the class
        :param kwargs:
        '''
        # Call builder of BoxLayout because we don't use default builder
        super(Storeopen, self).__init__(**kwargs)
        self.state = 'begin'
        Clock.schedule_interval(self.open_the_door, 1.0 / 45.0)

    def closeDoor(self, dt):
        '''
        Close store door.
        :param dt: refresh rate of schedule interval
        :return:
        '''
        self.delet.size[0] += 1
        self.doorBar.pos[0] = self.delet.size[0] - self.doorBar.size[0] / 2 + self.delet.pos[0] - 10

    def openDoor(self, dt):
        '''
        Open store door.
        :param dt: refresh rate of schedule interval
        :return:
        '''
        self.delet.size[0] -= 1
        self.doorBar.pos[0] = self.delet.size[0] - self.doorBar.size[0] / 2 + self.delet.pos[0] - 5


    def checkColide(self, obj1, obj2):
        '''
        Check if bars (door and store) collides with
        some limit switch.
        use obj1.collide_widget(obj2)
        :param obj1: bar.
        :param obj2: limit switch.
        :return: true if colides, false if not.
        '''
        return obj1.collide_widget(obj2)

    def open_the_door(self,dt):
        # initialize state machine
        if self.state == 'begin':
            # if door open
            # the white bar touch in the left white switch
            if self.checkColide(self.doorClose, self.doorBar):
                # initials "loop" of close door.
                Clock.schedule_interval(self.openDoor, 1.0 / 40.0)
                self.state = 'doorOpen'
        elif self.state == 'doorOpen':
            # if door is closed
            if self.checkColide(self.doorBar, self.doorOpen):
                # finish "loop" of close door.
                Clock.unschedule(self.openDoor)
                global openstore
                # Close the Popup screen
                openstore.dismiss()

class OutputForm(BoxLayout):
    '''
    Input form customer exiting form
    '''
    # Variables from KV file.
    exitphone = ObjectProperty()
    def exitinsert(self):
        '''
        Insert data to DB
        :return:
        '''
        # Using global variables to connect to the DB
        global host_k
        global user_k
        global password_k
        # Helps to find if the customer insert wrong input
        degel=None
        # Connecting to DB with worker host, username and password
        con = mdb.connect(host=host_k, user=user_k, passwd=password_k, db='maskup')
        cur = con.cursor()  # Making cursor
        if self.exitphone.text.isdigit(): # Check input
            cur.execute("SELECT customer_phone FROM customer WHERE out_hour IS NULL") # Quering DB - shows all phone numbers of customers that are currently in the store
            phonelist=cur.fetchall() # Retrive all rows from query
            # Running over all the phone numbers
            for p in phonelist:
                # Saving the phone number in variable
                phone_fix=str(p[0])
                # Checking if the phone number start with 0
                # Saving the phone number that the customer insert while he asked to exit the store
                # in different variable
                if self.exitphone.text[0]=='0':
                    checking_phone=self.exitphone.text[1:]
                else:
                    checking_phone = self.exitphone.text
                # If the number that was insert is equal to number in DB of customer that didn't exit the store yet
                if checking_phone == phone_fix:
                    degel=True
                    cur.execute("UPDATE customer SET out_hour = %s WHERE customer_phone = %s"
                                % (("'"+(str(datetime.now())[11:])+"'" ),checking_phone)) # Quering DB - updating the time that the customer exit the store
                    con.commit() # Makes transaction
                    con.close() # Close transaction
            # If the number that was insert isn't equal to number in DB of customer that didn't exit the store yet
            if degel==None:
                self.exitphone.text = 'Your number is not correct. Try again ' # Error msg
                return
        else:
            self.exitphone.text ='Please type valid data. Use 0-9 digits ' # Error msg
            return
        self.clear_widgets()  # Clears InputForm
        # Trigger the on_call_popup_maalitclose function
        self.on_call_popup_Storeclose()
        self.add_widget(Openscreen()) # Adds Openscreen as new widget

    def on_call_popup_Storeclose(self):
        '''
        Shows the Popup screen
        :return:
        '''
        global closestore
        # Creating Popup object that will show how the store's door closes
        closestore = Popup(title='     ', content=Storeclose(),background='inside.jpg')
        # Open the Popup screen and show it to the customer
        closestore.open()


    def back(self):
        '''
        Returns to the previous screen
        :return:
        '''
        self.clear_widgets()  # Clears OutputForm
        self.add_widget(Choose())  # Adds Choose as new widget

    def home(self):
        '''
        Returns to the home screen
        :return:
        '''
        self.clear_widgets()  # Clears OutputForm
        self.add_widget(Openscreen())  # Adds Choose as new form



class Storec(Widget):
    '''
    Store car
    '''
    pass
class Doorc(Widget):
    '''
    Store door
    '''
    pass

class DoorBarc(Widget):
    '''
    Door limit for door open, door close.
    '''
    pass
class Storeclose(Widget):
    '''
    Shows the store's door close
    '''
    # Variables from KV file.
    body = ObjectProperty
    delet = ObjectProperty
    doorBar = ObjectProperty
    doorOpen = ObjectProperty
    doorClose = ObjectProperty

    def __init__(self,**kwargs):
        '''
        Builder of the class
        :param kwargs:
        '''
        # Call builder of BoxLayout because we don't use default builder
        super(Storeclose, self).__init__(**kwargs)
        self.state = 'begin'
        Clock.schedule_interval(self.open_the_door, 1.0 / 45.0)


    def openDoor(self, dt):
        '''
        Open store door.
        :param dt: refresh rate of schedule interval
        :return:
        '''
        self.delet.size[0] -= 1
        self.doorBar.pos[0] = self.delet.size[0] - self.doorBar.size[0] / 2 + self.delet.pos[0] - 5


    def checkColide(self, obj1, obj2):
        '''
        Check if bars (door and store) collides with
        some limit switch.
        use obj1.collide_widget(obj2)
        :param obj1: bar.
        :param obj2: limit switch.
        :return: true if colides, false if not.
        '''
        return obj1.collide_widget(obj2)

    def open_the_door(self,dt):
        # initialize state machine
        if self.state == 'begin':
            # if door open and the car
            # the white bar touch in the left white switch
            if self.checkColide(self.doorClose, self.doorBar):

                # initials "loop" of close door.
                Clock.schedule_interval(self.openDoor, 1.0 / 40.0)
                self.state = 'doorOpen'
        elif self.state == 'doorOpen':
            # if door is closed
            if self.checkColide(self.doorBar, self.doorOpen):
                # finish "loop" of close door.
                Clock.unschedule(self.openDoor)
                global closestore
                # Close the Popup screen
                closestore.dismiss()


class maskup(App):
     'Runs main loop uses kv file name maskup'
     icon = 'surgical-mask-24.ico'
     pass

if __name__ == '__main__':  # Set entry point
    # maximized the kivy screen
    Config.set('graphics', 'fullscreen', 'one')
    Config.write()
    maskup().run()