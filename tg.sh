# tg.sh
#!/bin/bash
TOKEN="8584114284:AAGSa38ZQvFd2qUQSqqCpF-a22ZahzIh8FQ"
CHAT="6523358903"
curl -s -X POST \
  https://api.telegram.org/bot$TOKEN/sendMessage \
  -d chat_id=$CHAT \
  -d text="$1" >/dev/null
