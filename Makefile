vim:
	cp arcdsl-syntax.vim ~/.vim/syntax/arcdsl.vim
	cp arcdsl-ftdetect.vim ~/.vim/ftdetect/arcdsl.vim
	touch temp
	cat ~/.vimrc arcdsl.vimrc > temp
	mv temp ~/.vimrc
	mv arcdsl.vimrc arcdsl.vimrc.alreadyappended
