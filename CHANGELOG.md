# CHANGELOG

## v0.1.0 (2025-10-12)

### Chore

* chore: update files ([`a317db3`](https://github.com/bniladridas/rag/commit/a317db39685666a941e62e597c4e8db620f16c8e))

* chore: add .mypy_cache/ to .gitignore ([`e6ea8e3`](https://github.com/bniladridas/rag/commit/e6ea8e3f07ca0bd3a58b27c0cf57e76031c6b7ec))

* chore: remove deprecated files

- Remove scripts/rewrite_msg.sh (replaced by commit-msg hook)
- Remove src/rag_pipeline.py (deprecated wrapper)
- Remove src/sci_fi_dataset_collector.py (deprecated wrapper) ([`51a0814`](https://github.com/bniladridas/rag/commit/51a0814a07d3ebfb265ba444498719a42ade237a))

### Ci

* ci: add dependabot pr validation ([`9c93867`](https://github.com/bniladridas/rag/commit/9c938677e90687ab58bbda4b65043608b37db9ba))

* ci: update python version to 3.10 ([`cc4f4cb`](https://github.com/bniladridas/rag/commit/cc4f4cb1ed5dbf94e5149e585119d2b0d237c536))

* ci: add dependabot config ([`fe50fdd`](https://github.com/bniladridas/rag/commit/fe50fdd162dd6a3bdf650fad08a4b339ac663bfe))

* ci: add manual trigger to e2e workflow ([`fe40411`](https://github.com/bniladridas/rag/commit/fe404113909215f90f21ad4d37c6a246f42e0fce))

* ci: add e2e workflow ([`b245ec5`](https://github.com/bniladridas/rag/commit/b245ec597a9832a6733c81d7bebdc6772ad0da3d))

* ci: add github actions workflow

- add ci pipeline for automated testing
- run linting, import checks, and setup validation
- trigger on push/pr to main/master branches ([`213d9c8`](https://github.com/bniladridas/rag/commit/213d9c86c7d8478f6012c12ba1f0e82cf00f86b9))

### Documentation

* docs: update citation.cff ([`064ab58`](https://github.com/bniladridas/rag/commit/064ab58e7d18e7b6283e4c9532ac837d8684984d))

* docs: add citation.cff file ([`bf4c0ba`](https://github.com/bniladridas/rag/commit/bf4c0ba51d0a7a064f61068771b1abf2bcee893f))

* docs: add citations section ([`d9cec48`](https://github.com/bniladridas/rag/commit/d9cec48cbf0b6f6de7a047a38a188357521d04f7))

* docs: add codeowners file ([`b3a43a1`](https://github.com/bniladridas/rag/commit/b3a43a188faee972398e0d773238dcb6c428858b))

* docs: update readme structure ([`310b59e`](https://github.com/bniladridas/rag/commit/310b59eae910fff89f530fa0cb9efad9ed1beba0))

* docs: update docker references ([`daefc60`](https://github.com/bniladridas/rag/commit/daefc60198ba8e1f01878b49b0853bdd7cb3d4ee))

* docs: update api key config ([`5b22a35`](https://github.com/bniladridas/rag/commit/5b22a35e497509a665d56d5fbc7c706f2e35d809))

* docs: add dependabot section to readme ([`ff3673b`](https://github.com/bniladridas/rag/commit/ff3673b3a82d68bf9e484fbe48b8de045e7ac18e))

* docs: add security policy ([`82fa03b`](https://github.com/bniladridas/rag/commit/82fa03b7eba29554db22ed4ec56a61e97df8eaea))

* docs: add apache 2.0 license ([`5d7ee0d`](https://github.com/bniladridas/rag/commit/5d7ee0da213c40732b1723d0a61cea44c8a595d5))

* docs: ignore generated datasets in src/datasets ([`56d4454`](https://github.com/bniladridas/rag/commit/56d4454ef7686340ff2e5dcf2bd240c38f10d5a4))

### Feature

* feat: add semantic release bot ([`4a46f45`](https://github.com/bniladridas/rag/commit/4a46f4528f2faadb6d0292787c9f453c38f748f6))

* feat: add auto-fix lint script ([`88b9311`](https://github.com/bniladridas/rag/commit/88b93110cab5713a187469d8bff8b718af734003))

* feat: add test for data_fetcher ([`8e1700a`](https://github.com/bniladridas/rag/commit/8e1700a4fe57c94f0460e4173fbb3412c4ca5151))

* feat: add force tui flag for testing ([`c666dd2`](https://github.com/bniladridas/rag/commit/c666dd2a32f6bb8ce002a64bf839fef6e50338f1))

* feat: enhance ui integration tests ([`aca6929`](https://github.com/bniladridas/rag/commit/aca692922ec35a6a769ff020b67a452d9f3f006e))

* feat: add flake8 and config ([`ee6fab9`](https://github.com/bniladridas/rag/commit/ee6fab936060aa664a12fe6abacdafbb1d805299))

* feat: initial rag transformer project ([`5a7f27f`](https://github.com/bniladridas/rag/commit/5a7f27f2a5b416dc47c6f3662e5f71b5751b7615))

* feat: add tui test to e2e workflow ([`f692e75`](https://github.com/bniladridas/rag/commit/f692e75e81d0984c3ef2adf96a2f7eb2019a74bf))

* feat: add tui with rich and tests ([`227f1eb`](https://github.com/bniladridas/rag/commit/227f1eb3798fa68e100ae57dea153f6af7a49046))

* feat: add docker support and ci ([`bbed0fd`](https://github.com/bniladridas/rag/commit/bbed0fd691115af802eaad4a78a0ba23c2ce8a0e))

* feat: add rewrite script and docs ([`4200c1f`](https://github.com/bniladridas/rag/commit/4200c1fad767c262ffed3d2f14f3e1090f0f065d))

* feat: add unit and e2e tests with pytest ([`84d6429`](https://github.com/bniladridas/rag/commit/84d642905be6ee48ab6f220d8c147e5e24755fc6))

* feat: add docs, docker, and caching ([`6572576`](https://github.com/bniladridas/rag/commit/657257680f4eb002b3edc2f531bf0649fcf046ce))

* feat: add setup.py for packaging

- Enable pip install and distribution
- Add console scripts for CLI usage ([`8ba1a1f`](https://github.com/bniladridas/rag/commit/8ba1a1fbbf103a8425f5015ca50b8ee7b658e5b1))

* feat: add agentic tools

- Add WIKI tool for Wikipedia search
- Add TIME tool for current date/time
- Improve tool detection and execution
- Use Wikipedia REST API with proper headers ([`1bb2f11`](https://github.com/bniladridas/rag/commit/1bb2f11452a8c71420a857984cda16a3ead5c981))

* feat: setup conventional commits ([`092cc3c`](https://github.com/bniladridas/rag/commit/092cc3c2abe94bfbcc0f5b6d598ab09a2005ed7d))

### Fix

* fix: install wheel for build ([`564bba1`](https://github.com/bniladridas/rag/commit/564bba1348fdc8a2fb1a4521a5f1f751d9a3a43d))

* fix: use harper release bot token ([`a4f907f`](https://github.com/bniladridas/rag/commit/a4f907f726aeae228366c6e814c507c633f4fcc4))

* fix: update release workflow ([`d03b348`](https://github.com/bniladridas/rag/commit/d03b3480a88504e469737f6752ed330eb88d8099))

* fix: remove emoji from security policy ([`7302d3a`](https://github.com/bniladridas/rag/commit/7302d3afd5dccd94ecacaa889f81117eae555774))

* fix: remove emojis from scripts ([`cf43502`](https://github.com/bniladridas/rag/commit/cf435027cdc8023af5781f0a3a904a9c65035cbd))

* fix: apply linting fixes ([`12a5345`](https://github.com/bniladridas/rag/commit/12a534513859797985458fb97158591d00df44c4))

* fix: update tui test patches ([`dc92695`](https://github.com/bniladridas/rag/commit/dc9269541986bb97a59623863a3a90f5683f54f9))

* fix: remove package install from ci ([`66345cc`](https://github.com/bniladridas/rag/commit/66345ccc22f5b6ac695d551826e8e01ee6fb5a9a))

* fix: reformat yaml indentation ([`5709227`](https://github.com/bniladridas/rag/commit/570922713d5774d8c44c80677fa8a2b444dbeba8))

* fix: yaml indentation in ci ([`ee673b1`](https://github.com/bniladridas/rag/commit/ee673b185b845acc11b58a60aa91c70dd18205e7))

* fix: add pythonpath to test run ([`a173a70`](https://github.com/bniladridas/rag/commit/a173a70423287b849c982dd832a2c8b383a28f2b))

* fix: install package in ci ([`b39e678`](https://github.com/bniladridas/rag/commit/b39e678c1be38687295971692681e803259f7208))

* fix: add pythonpath to e2e test run ([`880d68c`](https://github.com/bniladridas/rag/commit/880d68c6dfbdb1c420e10c143b84fa8710155997))

* fix: set pythonpath in docker for src ([`9d3738a`](https://github.com/bniladridas/rag/commit/9d3738ae849d586488d7a466c75e044874aa78cc))

* fix: lint issues in __init__.py ([`adb9387`](https://github.com/bniladridas/rag/commit/adb9387142758da55f3278d5280ac24ab33e1b63))

* fix: add main to rag __init__.py ([`f5f2685`](https://github.com/bniladridas/rag/commit/f5f26859c8b53a5f6a73d19dc61a74626b9fd898))

* fix: tui test assertions ([`c46ad8a`](https://github.com/bniladridas/rag/commit/c46ad8a0de4c130af5e06fced04792365d2a161d))

* fix: config test assertion for path ([`7ad63ca`](https://github.com/bniladridas/rag/commit/7ad63cafb2aa779f22dc42c0bd9b313659a3627d))

* fix: correct imports in unit tests ([`dc04f16`](https://github.com/bniladridas/rag/commit/dc04f1671b149c9c730aa23ec45e5c24bf77cccb))

* fix: remove .coverage from repo ([`a16f88b`](https://github.com/bniladridas/rag/commit/a16f88b27015ecbfc70f66a797821f456fd5afd5))

* fix: correct tui import in e2e ci ([`56efebd`](https://github.com/bniladridas/rag/commit/56efebde3133941a0833ada58bbee13ffee3e228))

* fix: add readme.md copy in dockerfile ([`88dfc7f`](https://github.com/bniladridas/rag/commit/88dfc7fb892ddb131340fa95b3e2c57a3c4e94e5))

* fix: add config attributes and fix lint ([`e407269`](https://github.com/bniladridas/rag/commit/e407269078a008515352717e43f9571911423f09))

* fix: lint errors and setup entry points ([`5646180`](https://github.com/bniladridas/rag/commit/56461801d0de7712a1c5d8cba3ee1276dfe2bd07))

* fix: e2e test imports and patches ([`6118c0f`](https://github.com/bniladridas/rag/commit/6118c0f3416fc660994cf63c8dd029feefb3476e))

* fix: test before prune in ci ([`8318275`](https://github.com/bniladridas/rag/commit/8318275f39e1f1a94249c8037cc82f828a8b1d1b))

* fix: yaml indent fix ([`65b3902`](https://github.com/bniladridas/rag/commit/65b39020c4f52ff9a4283610daa5d90a9871b7ee))

* fix: yaml indentation in docker.yml ([`d2293e1`](https://github.com/bniladridas/rag/commit/d2293e1c278beca1a2b00fb7d919070dd17102b8))

* fix: ci disk space optimization ([`e6b1847`](https://github.com/bniladridas/rag/commit/e6b1847c393239ef37a47b250e5ab5156f63c661))

* fix: lint errors in tui.py ([`811ed17`](https://github.com/bniladridas/rag/commit/811ed1786637cab02dac4e6f370cbe5ccd4ce3e8))

* fix: remove tui e2e test from ci ([`971dfaf`](https://github.com/bniladridas/rag/commit/971dfaf65789d76e46699a11b804285aba3b0567))

* fix: use python -c for tui e2e test ([`5d96783`](https://github.com/bniladridas/rag/commit/5d967835addd8828bd6acdc1d5e1685770fb9e49))

* fix: disable model preload for ci ([`64abc50`](https://github.com/bniladridas/rag/commit/64abc50c6d62e2f0456758e3d56f92bbef0ee320))

* fix: model preload with app user perms ([`b732813`](https://github.com/bniladridas/rag/commit/b732813705761623fa5bee4e8162e4932e025b81))

* fix: optimize docker for disk space ([`acaad8b`](https://github.com/bniladridas/rag/commit/acaad8bff41513a9c606251364469ff25eae111e))

* fix: load docker image for testing ([`2e862e2`](https://github.com/bniladridas/rag/commit/2e862e28cb95cd466ea6dfbe163e9e3d4786d101))

* fix: correct docker tag and cli ([`15099c8`](https://github.com/bniladridas/rag/commit/15099c82d7e27fb2c7c7d3fc89bdedf461b411e7))

* fix: remove unused import and long line ([`a08a76b`](https://github.com/bniladridas/rag/commit/a08a76bed092a6c6828488cabd0c246bdccea5af))

* fix: add type ignore for faiss ([`91fd15e`](https://github.com/bniladridas/rag/commit/91fd15eed8a4c983e30688b07f1736e67912d98e))

* fix: update dockerfile python version ([`af4cf7b`](https://github.com/bniladridas/rag/commit/af4cf7b8d3651b1cfdfcc90a83c9b72182883b5c))

* fix: update package versions ([`ddc0aa5`](https://github.com/bniladridas/rag/commit/ddc0aa52c410fcb9d6e20b6be84dbde782c2ec57))

* fix: update all packages to latest ([`b84b972`](https://github.com/bniladridas/rag/commit/b84b972e3a9bff5791fc1f4a38c47685137c5aeb))

* fix: upgrade setuptools in ci ([`e7eb98b`](https://github.com/bniladridas/rag/commit/e7eb98b51d6a21acdf6cc802dc441cb9af76b0d4))

* fix: update faiss-cpu to 1.8.0 ([`5ccfaf3`](https://github.com/bniladridas/rag/commit/5ccfaf3599167ed8d842524df249f3af6338bc3b))

* fix: update torch to 2.2.0 ([`12c2949`](https://github.com/bniladridas/rag/commit/12c29494cfdfddcc2cdca9439f9108f83ca3951f))

* fix: correct path in test script ([`f9684bb`](https://github.com/bniladridas/rag/commit/f9684bb571325d3db677e565d97346df13d1b6a8))

* fix: use abspath in lint.py ([`7f08a14`](https://github.com/bniladridas/rag/commit/7f08a148255a6c4124268e6dacce0650e45a14ce))

* fix: update transformers to 4.46.3 ([`861d594`](https://github.com/bniladridas/rag/commit/861d5943c349b890d420d5a334ad6a9a7bc27aec))

* fix: use relative imports for packaging

- Update all imports to relative imports (.module)
- Fix setup.py entry points with correct paths ([`928f2a1`](https://github.com/bniladridas/rag/commit/928f2a168cfcaaf17982591cdfd8fd91a829fff9))

* fix: improve tool handling

- Detect CALC: queries directly
- Allow math functions like sqrt in calculations
- Strip quotes from input queries ([`f86930c`](https://github.com/bniladridas/rag/commit/f86930c6e714ea0f8144ea3b79d41ddefbdf0efb))

* fix: add direct calculation handling

- Detect calculation queries and compute directly
- Avoid relying on model to use CALC tool ([`9e1c237`](https://github.com/bniladridas/rag/commit/9e1c2373762a824deb4af06884d56220b83b1814))

* fix: improve data collector reliability

- Add retry logic with exponential backoff
- Use requests session for connection reuse
- Improve error handling for API calls
- Fix dotenv loading path ([`708b6db`](https://github.com/bniladridas/rag/commit/708b6db48309bd7639b952256a58bf34646ba418))

* fix: use requests for tmdb api ([`95f4b5a`](https://github.com/bniladridas/rag/commit/95f4b5aab662756ac44dee3547306f5fba97de0e))

* fix: update dataset loading path to src/datasets ([`8bbb794`](https://github.com/bniladridas/rag/commit/8bbb7943019a768c39741b0b63147ed6dfe4a39e))

### Refactor

* refactor: move ui into rag package ([`3c355b5`](https://github.com/bniladridas/rag/commit/3c355b50779d8c2b761a60b2a2552f8ba916612a))

* refactor: move e2e test to integration ([`59accda`](https://github.com/bniladridas/rag/commit/59accdaa1d5fdfcfe99694988cab8804459c5777))

* refactor: move docker files ([`bc35347`](https://github.com/bniladridas/rag/commit/bc353474c045ad26fa3cb63977e0d88995a437f3))

* refactor: organize docker files ([`9305833`](https://github.com/bniladridas/rag/commit/9305833d7648328e6779d80ef6a7062ce3f21487))

* refactor: rename to rag ([`99a9f36`](https://github.com/bniladridas/rag/commit/99a9f36847716934d8ce4899fb2457c6e1f11c64))

* refactor: modularize into parallel files

- Split into config, data_fetcher, tools, rag_engine, main
- Create package structure with setup.py
- Cleaner organization while maintaining functionality ([`2d1b229`](https://github.com/bniladridas/rag/commit/2d1b22905d79889ae0332d980bef5b1dacfc4948))

* refactor: parallel data fetching

- Remove hardcoded knowledge
- Fetch data from APIs in parallel
- Cleaner code architecture
- Improved RAG responses ([`209a040`](https://github.com/bniladridas/rag/commit/209a040a450b6b2d365543b4665ee3efd4714a85))

### Style

* style: break long tokenizer call ([`041a3d2`](https://github.com/bniladridas/rag/commit/041a3d218d6fc36511951c1253446df0ba0d8243))

* style: fix long lines in rag_engine.py ([`68e6e37`](https://github.com/bniladridas/rag/commit/68e6e3793ba2c8dcb55f5dbb73e214f4b4001298))

* style: fix remaining linting issues ([`6c56b98`](https://github.com/bniladridas/rag/commit/6c56b98d8c2da1dc88b56aa6716b619db89dab61))

* style: fix linting issues (flake8, mypy) ([`6005266`](https://github.com/bniladridas/rag/commit/6005266328d3ffdc3962f9d48e345667f8f1591c))
