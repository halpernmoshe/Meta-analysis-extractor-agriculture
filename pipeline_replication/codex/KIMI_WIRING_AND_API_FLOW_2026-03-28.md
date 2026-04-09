## Kimi wiring and API flow

### Connection status

The key saved in:

- [`.env.kimi`](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\.env.kimi)

was verified successfully against the official Moonshot base URL:

- `https://api.moonshot.ai/v1`

A live authenticated `models.list()` test returned `kimi-k2.5` and other current models.

### Wiring changes made

The following files now load `.env.kimi` as a fallback after `.env`:

- [config.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\config.py)
- [consensus_pipeline.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\consensus_pipeline.py)
- [modules/kimi_figure_extract.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\modules\kimi_figure_extract.py)

This means:

- normal shell env vars still win
- `.env` still works
- `.env.kimi` now works without copying the key into `.env`

### Smoke tests completed

1. Config import test
- `config.MOONSHOT_API_KEY` loads successfully

2. Kimi extractor initialization
- `KimiFigureExtractor(mode='instant')` initializes successfully

3. Live chat completion test
- one small `chat.completions.create()` call to `kimi-k2.5`
- prompt: `Reply with exactly the word OK.`
- response: `OK`

### Current Kimi call sites in the repo

#### 1. Text extraction

Main function:

- [extract_kimi()](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\consensus_pipeline.py#L1557)

Actual API call:

- [chat.completions.create()](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\consensus_pipeline.py#L1580)

What it sends:

- model: `kimi-k2.5`
- system message containing paper text
- user message containing the unified extraction prompt
- `extra_body={"thinking": {"type": "enabled"}}`

This is one text extraction call per paper.

#### 2. Figure extraction

Main function:

- [extract_figure_from_image()](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\modules\kimi_figure_extract.py#L299)

Actual API call:

- [chat.completions.create()](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\modules\kimi_figure_extract.py#L318)

What it sends:

- model: `kimi-k2.5`
- text prompt
- one base64 PNG as `image_url`
- `extra_body={"thinking": {"type": "enabled"}}`

This is one multimodal call per page/image processed.

#### 3. Batched vision path inside consensus pipeline

Additional Kimi vision call site:

- [consensus_pipeline.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\consensus_pipeline.py#L1947)

What it sends:

- model: `kimi-k2.5`
- one user message with multiple `image_url` items plus a text prompt
- `extra_body={"thinking": {"type": "enabled"}}`

This is the best place to batch figure pages if we want fewer API calls.

### What a 10-paper run would do

If we run text extraction only:

1. load topic config / selected papers
2. extract or load PDF text per paper
3. run one Kimi text extraction call per paper
4. parse JSON
5. save observations, token counts, and cost estimates

That is roughly:

- 10 `POST /chat/completions` calls

If we also run figure extraction:

1. identify papers/pages needing figure reading
2. render relevant pages to images
3. send one or more multimodal calls per paper or page batch
4. parse figure JSON
5. merge figure observations into the extraction result

That adds:

- anywhere from a few to many extra `POST /chat/completions` calls depending on batching

### Practical recommendation for the first 10-paper prototype

Do it in two passes:

1. Text-first run on all 10 papers
- cheaper
- easier to debug
- gives baseline extraction output

2. Figure-only rescue pass on the subset that clearly needs it
- lower cost
- avoids spending vision calls on easy table papers

### Context caching

Moonshot does have a context caching feature in official blog posts, including:

- https://platform.moonshot.ai/blog/posts/context-caching
- https://platform.moonshot.ai/blog/posts/how-to-save-90-percent-with-context-caching

Important caution:

- the official current chat docs clearly document the main `chat.completions` path
- the caching material I found is in blog posts and older examples
- I have not yet verified current production docs for `kimi-k2.5` caching semantics
- the repo does not currently implement Moonshot caching

So for now:

- assume no caching in the first prototype
- treat caching as a second-step optimization

### Where caching would matter most

Caching is probably most useful if we do repeated full-context calls on the same paper, for example:

- multiple role-agents all reading the same full paper
- repeated rescue / adjudication calls on the same paper text
- repeated benchmark-comparability passes

Caching matters less if we only do one extraction call per paper.

### Best next step

Before the 10-paper batch:

1. choose the 10 papers
2. run 1 paper end-to-end with Kimi text extraction
3. inspect JSON quality and token usage
4. then scale to 10

