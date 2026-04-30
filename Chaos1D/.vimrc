call plug#begin('~/.vim/plugged')
Plug 'JuliaEditorSupport/julia-vim'
Plug 'christoomey/vim-tmux-navigator'
call plug#end()

filetype plugin indent on
set tabstop=4
set shiftwidth=4
set expandtab
set number
set wrap
set hlsearch
syntax on

nnoremap <C-L> <C-W><C-L>
nnoremap <C-H> <C-W><C-H>
nnoremap <C-K> <C-W><C-K>
nnoremap <C-J> <C-W><C-J>
tnoremap <C-L> <C-W><C-L>
tnoremap <C-H> <C-W><C-H>
tnoremap <C-K> <C-W><C-K>
tnoremap <C-J> <C-W><C-J>

inoremap ji <Esc>
nnoremap ,; :update<cr>
nnoremap m x
vnoremap m x
nnoremap gh dd

let g:tmux_navigator_no_mappings = 1
noremap <silent> <c-h> :<C-U>TmuxNavigateLeft<cr>
noremap <silent> <c-j> :<C-U>TmuxNavigateDown<cr>
noremap <silent> <c-k> :<C-U>TmuxNavigateUp<cr>
noremap <silent> <c-l> :<C-U>TmuxNavigateRight<cr>


