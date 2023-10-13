##  end to end machine learning  
6145693007:AAHS4gQg8Lku1WP_qobU69_D1KtBp5B68nk 28642796 387679e2ce84039bd7fe11b455de5dd4

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
6412385940:AAFUHGCmfZOtcb8fh4bldXpbHnQrLlDX4F4

### Clone
git clone  https://github.com/arita37/myutil.git
cd myutil
git checkout devtorch


### Installl in Dev mode
cd myutil
pip uninstall utilmy
pip install -e  .         ### Dev mode install

python -c "from utilmy import log, date_now"



Do not update the repo globally, only one file change.

Tasks :
1) Start using utilmy

2) Add some new tests.

3) Only after adding tests, you are authorized to create new functions.

Please do a pull request when change to the code


###  Check if your code passed tests
https://github.com/arita37/myutil/actions/workflows/aa_build_test_only.yml


### Github actions code
https://github.com/arita37/myutil/blob/devtorch/.github/workflows/aa_build_test_only.yml


#### Docs:
https://arita37.github.io/myutil/en/zdocs_y23487teg65f6/utilmy.webscraper.html#module-utilmy.webscraper.cli_reddit


#### Entry files are 
utilmy/utilmy_base.py
utilmy/oos.py




