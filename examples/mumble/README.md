# LibreASR Simple Mumble Bot

First install the required dependencies:

```bash
bash ./examples/mumble/install.sh
```

Then, running

```bash
python3 -m examples.mumble mumble.example.com --channel "Talk #1" --nickname "LibreASR" --users "joe,mike"
```

will load a `LibreASR` instance, connect to the
[Mumble](https://www.mumble.info/) server at
`mumble.example.com`, join the channel `Talk #1`,
listen for speech by the users `joe` and `mike` and
print live transcripts to the channel chat :rocket:

Extra arguments:

```
usage: python3 -m examples.mumble [-h] [--nickname NICKNAME] [--password PASSWORD]
                   [--channel CHANNEL] [--users USERS] [--dry-run DRY_RUN]
                   [--lang LANG] [--conf CONF]
                   server

positional arguments:
  server                URL of Mumble server to use

optional arguments:
  -h, --help            show this help message and exit
  --nickname NICKNAME   Bot Nickname of Mumble server to use
  --password PASSWORD   Mumble server password
  --channel CHANNEL     Mumble channel name to join
  --users USERS         Users LibreASR should listen to ('all', 'joe,mike')
  --dry-run DRY_RUN     Don't connect to Mumble, just load LibreASR
  --lang LANG           Language to use ('en', 'de', ...)
  --conf CONF, --config CONF
                        Path to LibreASR configuration file
```