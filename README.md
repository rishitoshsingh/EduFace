# EduFace Documentation

## Multiprocessing

### Creating custom process

For this we need to understand how Process class works and which method should we override

[Docs](https://docs.python.org/3/library/multiprocessing.html#reference)

### Message passing between process

queue.Queue can't be used for multiple Process. For this multiprocessing provides its own queue(s). From which I have used JoinableQueue just because they can be joined (if we need to terminate, then program should wait till all items in queue are processed)

[Docs](https://docs.python.org/3/library/multiprocessing.html#pipes-and-queues)

### Sychronization between processes

Synchronization is needed between process. First Consumer is started, all other producers process should wait till Consumer is ready to process frames (loading models in GPU takes time)

For this I have used multiprocessing.Event which is same as threading.Event

[Docs](https://docs.python.org/3/library/multiprocessing.html#synchronization-primitives)

## Running EduFace as system service

For this companies/developers earlier used to create service using crontab. Now every softwares for linux is handled through systemd. With systemd it is easy to handle services. We can send TERM signal using systemctl. These signals can be handled through signal module in python

To create systemd service, we need to create *.service file. We can also restrict hardware usage using *.slice file (For EduFace, I have restricted RAM to 12 GB ). Restricting hardware resource was not possible through crontab.

After creating *.service files, we have to move them to /etc/systemd/system/

To start/stop/restart service

```console
foo@bar:~$ sudo systemctl start/stop/restart *.service
```

To check status of service

```console
foo@bar:~$ sudo systemctl status *.service
```

To view all service log 

```console
foo@bar:~$ sudo journalctl -u *.service 
```
To view last lines of log
```console
foo@bar:~$ sudo journalctl -u *.service -f
```



To start service at every boot:

```console
foo@bar:~$ sudo systemctl enable *.service
```

[Docs 1](https://www.digitalocean.com/community/tutorials/understanding-systemd-units-and-unit-files), [Docs 2](https://www.freedesktop.org/software/systemd/man/systemd.service.html)
