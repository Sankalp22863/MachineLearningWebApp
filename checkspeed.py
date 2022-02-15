# import speedtest

# test = speedtest.Speedtest()

# print('Loading the server list...')
# test.get_servers() # -> Get list of all the servers avaialbale for speedtest.
# print("Choosing the best server...")
# best = test.get_best_server() # -> Choose best Server.

# print(f"Found {best['host']} located in {best['country']}.")



import speedtest
wifi  = speedtest.Speedtest()
print("Wifi Download Speed is ", wifi.download())
print("Wifi Upload Speed is ", wifi.upload())


