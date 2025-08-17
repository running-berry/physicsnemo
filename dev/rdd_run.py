#This code will run in 3hr after it is started  
import time
import logging

start_time = time.time()
flag = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Process started.")

while(flag):
   #if time.time() - start_time > 10800:
   if time.time() - start_time > 100:
       print("passed, stopping the process.")
       flag = False
       break
   time.sleep(60)  # Check every minute

logging.info("Process ended.")