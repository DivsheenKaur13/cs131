# Quick jump into my cs131 repo
alias cs131='cd ~/cs131'

# Cleaner git log for checking commit history
alias gl='git log --oneline --graph --decorate'

# Make a new worksheet folder and cd into it, but warn me if it already exists
mkws ()
{
	if [ -d "$1" ]; then
		echo "$1 already exists, not overwriting it"
	else
		mkdir -p "$1"
		cd "$1"
	fi
}
