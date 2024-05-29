---
layout: post
title: "Github Blog 제작일기"
subtitle: "[Tips]"
date: 2024-05-29 20:54
background: 
tag: [Tips, Github io, Notion]
---

# 루시의 깃헙블로그

길고 길었던 나의 깃헙블로그 만든 과정을 기록해보려고 한다.

만들기 시작할 때는 1시간이면 만들겠지 싶었는데 점점 블로그가 심해로 빠지더니 3시간씩 끊어서 3일 총 9시간이 걸렸다. 중간에 삽질을 엄청나게 했다. 삽질 할 때마다 느끼지만 계란으로 바위 치는 느낌이다. 삽질도 하면서 배우는 게 생기므로 뭐.. 이제 체념하고 묵묵히 어떻게든 되겠지 하면서 삽질을 한다. 그럼 신기하게 언젠가 어떻게든 되긴 된다. 

이번 삽질의 주요 원인은 무지성 따라하기였다. 처음에 다른 블로그 글들을 보면서 따라했다. 처음부터 끝까지 그대로 따라했으면 차라리 금방 성공했을텐데 난 괜히 내 맘에 드는 테마를 사용하고 싶어서 글에서 사용한 테마 말고 다른 테마를 찾아서 썼더니 이유를 모르겠는 에러가 계속 났다. 코드를 이해하고 따라한 게 아니여서 도저히 어디서 에러가 나오는지 알 수 없었다.. 무지성 따라하기 지양하자

그리고 이번에 처음으로 에러 난 부분을 하나하나 읽으면서 필요한 부분만 구글에 검색하고 코드를 뜯어봤다. 에러난 부분을 GPT에 무작정 복붙하지 않았다. 폴더 깊숙히 있는 파일 코드 들어가서 고쳐보면서 뭐가 어떻게 돌아가는지 알 수 있었다. 

암튼 전체적인 나의 **타임라인**은

1. 리포지토리 생성
2. 1트: Hexo 사용 시도 후 포기
3. 2트: Ruby, Jekyll, Bundler 설치하다가 버전 안 맞아서 다 밀어버림
4. 3트: Jekyll 테마 1 적용, html 파일 인식 실패
5. 4트: Jekyll 테마2 적용, 레이아웃 오류
6. 5트(막트): Jekyll 테마3 적용 성공 → 블로그 생성 완료!!

와 같이 진행되었다.. 첫 번째 스타트~~

## 1. 리포지토리 생성

![Untitled](/_publications/240529/5ef2fedf-6ec5-49a3-a3dc-f4664e55d8d4.png)

먼저 새로운 리포지토리를 만들어준다. 리포지토리 이름은 ’자기깃헙ID’+’.github.io’ (ex> Lucypothesis.github.io)로 설정해야 한다. 밑에 Add a README file만 체크해주고 그대로 create repository 하면 된다.

## 2. 1트: Hexo 사용 시도 후 포기

깃헙 테마툴에는 Jekyll과 Hexo, Hugo 등이 있다. 처음에는 Hexo를 사용했다. 이유는 딱히 없다. 그냥 Jekyll이 더 많이 사용되는 것 같아서 괜히 Hexo를 사용해보고 싶었다. (여기서부터 잘못되었다. 사람들이 많이 쓰는 데는 다 이유가 있다.)

![Untitled](/_publications/240529/Untitled.png)

왼쪽과 같이 계속 WARN No layout이 뜨면서 [localhost:4000](http://localhost:4000) 에도 아무것도 안 나왔다. 아마 진짜 html 파일이 비어있었던 것으로 추정된다. 직접 html을 만들 자신이 없어서 그냥 포기했다. 여기까지 하는데 3시간 씀 (삽질하면서 느낀점: **초보이고 코드가 이해가 안 되는데 빨리 끝내고 싶다면 그냥 사람들이 많이 쓰는 툴 쓰자**)

## 3. 2트: Ruby, Jekyll, Bundler 설치하다가 버전 안 맞아서 다 밀어버림

Hexo를 포기하고 Jekyll을 쓰기로 했다. Jekyll을 사용하려면 Ruby를 설치해줘야 한다. Jekyll이 Ruby로 만들어졌고 Ruby의 런타임 환경에서 실행되기 때문이다. 그래서 순서를 Ruby 설치 → Jekyll, Bundler 설치로 가야 한다.

먼저 Ruby 홈페이지에 들어가서 Ruby를 설치해준다. 설치 막바지에 오른쪽 화면이 보일텐데 여기서 꼭 1 enter 하고 3 enter까지 해야 Ruby가 제대로 설치된다. 처음에 영어길래 대충 읽고 esc 눌렀다가 Ruby 설치가 안 돼서 시간 2배로 씀… **영어로 나와 있다고 귀찮아하지 말고 잘 읽자.** 해결책은 눈앞에 있다. 단지 영어일 뿐….

![Untitled](/_publications/240529/Untitled%201.png)

Jekyll 설치는 터미널에서 `gem install jekyll` 해주면 된다. Jekyll은 html을 정적으로 렌더링 해준다고 한다. 왜 Jekyll을 썼냐고 물어보면 할 말이 없다. 사람들이 많이 쓰길래 썼다. 근데 사람들이 왜 많이 쓸까? 뭐 뭐가 편리하고 깃헙에 자동으로 연동된다는 설명을 읽었지만 솔직히 잘 이해가 가진 않는다.. 나중에 이 글을 다시 읽을 때쯤이면 뭔가 경험적으로 알게 되려나? 하하

Bundler 설치도 터미널에서 `gem install bundler` 해주면 된다. Bundler는 의존성 관리 도구이다. RubyGems를 통해서 필요한 라이브러리(gem) 버전을 설치하고 관리하는 역할을 한다고 한다. 

지금 Ruby의 최신 버전은 3.3.1 이긴 한데 뭐 하다가(기록 안 해놔서 기억 안 남) Ruby 2.7 버전이 필요하다고 해서 3.3 uninstall하고 2.7로 다시 설치했었다. 각 테마에 맞는 Ruby랑 Bundler 버전 맞추는 게 진짜 까다롭다. 뭐를 하면 Ruby 버전이 안 맞고 새로 설치하면 Bundler 버전이 안 맞고…

![Untitled](/_publications/240529/Untitled%202.png)

![Untitled](/_publications/240529/Untitled%203.png)

이때부터 심해로 빠진다는 느낌이 들었다. 별거 아닌 거 가지고 고통받는 느낌.. 버전때매 고통받아서 걍 Ruby랑 bundler 다 uninstall하고 깃헙 블로그까지 다 밀고 새로운 마음가짐으로 다시 하기로 함 ^^

## 4. 3트: Jekyll 테마 1 적용, html 파일 인식 실패

자 다시 새로운 마음으로 시작하자자. 새로운 마음에 필요한 건 새로운 테마! 그래서 [http://jekyllthemes.org/](http://jekyllthemes.org/) 에서 아래 그림과 같은 jekyll 테마를 찾았다.( [https://github.com/aidewoode/jekyll-theme-mint](https://github.com/aidewoode/jekyll-theme-mint))

![Untitled](/_publications/240529/d8a7e833-ec6e-478a-b14b-3339e6a3ba39.png)

깔끔 그 자체다. 글씨체, 색조합, 레이아웃 모두 마음에 들었다. 요놈이다 싶었다. 근데.. 

![Untitled](/_publications/240529/Untitled%204.png)

로컬로 돌려보니까 html이 적용이 안 되는 것이다. 아 또 왜 시험에 들게 만들어.. 하면서 오류를 읽어봤다.

![Untitled](/_publications/240529/Untitled%205.png)

assets\dark.scss 파일에서 expected ‘{’ 오류가 났다고 한다. dark.scss 파일에 들어가봤다. 근데 정말.. 내가 css를 겉핡기식으로만 알고 있기 때문에 뭐가 잘못된 지를 모르겠었다. 밑에 있는 저 파일에서 주석 뺀 상태가 끝이었다. GPT의 도움을 받았다. GPT 왈:

![Untitled](/_publications/240529/5416dcde-7a3f-4386-9f71-399c234d1791.png)

라고 한다. _sass 폴더에 파일을 저장하면 assets 폴더 안의 파일에서 @import 를 통해 참조된다. _sass폴더와 assets폴더의 존재 이유와 둘 사이의 관계를 알게 되었다. 

![Untitled](/_publications/240529/08a8f496-d4a5-4c36-9965-52df21ea92fe.png)

![Untitled](/_publications/240529/Untitled%206.png)

그래서 뭐가 잘못된거지 싶었다. 뭐라도 해보기로 했다. 일단 @import 에 있는 빨간 줄을 없애는데 집중해보기로 했다. 위에 ‘—-’ 부분을 주석 처리 했더니 @import 에 있던 빨간 줄이 없어졌다. 저 부분이 문제였구나! 옳다구나 에러 찾았다 싶었는데… 저 부분을 지우면 안 된다고 한다. GPT왈, ‘—-’를 front matter 라고 부르고, Jekyll은 프론트매터가 있는 파일을 정적 파일이 아니라 처리해야 할 파일로 간주한다고 한다. 근데 이 부분을 주석처리 해버리게 되면 `@import ‘dark’;` 만 남기 때문에 내용이 없는 것처럼 보일 수 있다고 한다. ‘—-’ 부분이 꼭 필요한 부분이었구나 싶었다..

![Untitled](/_publications/240529/Untitled%207.png)

그래서.. _sass 폴더에 있는 dark.scss와 assets폴더에 있는 dark.scss가 이름이 똑같아서 생기는 재귀적 문제인가 싶어서 assets 폴더에 있는  dark.scss → dark_theme.scss로 바꾸고 light.scss → light_theme.scss로 바꿔봤는데도 해결되지 않았다. 이름의 문제가 아니었던 것 같다. 바로 이름 원상복귀 시켰다. 

![Untitled](/_publications/240529/Untitled%208.png)

이게 에러를 바로바로 기록을 안 해놓으니 블로그도 중구난방으로 간다. 시간 순서가 기억이 안 난다. 그래서 그냥 캡쳐해놓은 에러 2가지를 설명하자면, 

![Untitled](/_publications/240529/Untitled%209.png)

![Untitled](/_publications/240529/Untitled%2010.png)

**에러1**: 기존에 하다가 Gemfile.lock 가 생성되어있는 상태라면 bundle install 이 작동하지 않는 것도 깨달았다. Gemfile만 있는 상태로 bundle install를 하면 이에 따른 Gemfile.lock가 생성되는 것 같았다. 그래서 bundle install의 기능이 궁금했다. GPT 왈 bundle install을 하면 Gemfile.lock를 업데이트 하고 Gemfile에 써있는 새로운 gem들을 설치한다고 한다. 윈도우라면 wdm도 설치한다고 한다. 

![Untitled](/_publications/240529/Untitled%2011.png)

![Untitled](/_publications/240529/Untitled%2012.png)

**에러2**:  `bundle install`을 하면 계속 이 부분에서 오류가 나는 것이다! 뭐지.. 하면서 에러문구를 봤다. 저 경로에 있는 rb_monitor.c 파일에서 오류가 나는 거라고 하는데 나는 C언어를 몰라서 GPT에게 파일 복붙해서 물어봤다.  그랬더니 `#include “ruby/thread.h”` 를 추가하라고 해서 추가했다. 그런데도 안 됐다. 역시 무지성으로 GPT한테 물어보면 해결은 안 되고 코드가 산으로 감을 다시 느낄 수 있었다. 물어볼 때도 알고 물어봐야 한다!

## 5. 4트: Jekyll 테마2 적용, 레이아웃 오류

아 걍 이 테마 자체를 포기하기로 했다. 다른 테마를 찾기로 했다. 그래서 찾은 2번째 테마 

[How to install notetheme](https://dinhanhthi.github.io/notetheme/how-to-install-notetheme)

얘도 적용해봤더니 [Lucypothesis.github.io](http://Lucypothesis.github.io) 로 들어가지긴 하는데 레이아웃이 이상하게 나왔다. 메인 페이지가 사이드바 아래로 가려졌었다. 캡쳐는 안 해놓음.. 나는 이걸 조정할 자신이 없었고 막상 적용해보니 묘하게 테마가 내 스타일이 아니었기 때문에 과감히 밀어버리기로 했다.

## 6. 5트(막트): Jekyll 테마3 적용 성공 → 블로그 생성 완료!!

지금까지 두 테마 모두 [jekyll theme org](http://jekyllthemes.org/) 에서 찾았다. 근데 여기가 오픈소스 페이지다보니.. 필요한 요소가 없는 에러가 계속 난다고 판단했다. 그래서 무조건 **`github.io` 사이트가 있는 테마** (∵ 사이트가 존재하면 테마가 잘 적용된다는 보증이 됨)를 탐색하기로 했다.

그래서 github 검색창에 `jekyll-theme`으로 검색해서 star가 많이 박혀있는(검증된) 테마 중 가장 마음에 드는 거를 선택했다. 세부적인 요소는 내가 고쳐서 쓰기로 했다. 많이 포기했다.

그래서 발견한 게 아래 사이트이다. ([https://academicpages.github.io/](https://academicpages.github.io/))

![Untitled](/_publications/240529/Untitled%2013.png)

나름 미니멀하고 깔끔하고 위에 탭도 있어서 조정해서 쓰기 괜찮아 보였다. 그리고 가장 매력적이었던 건! 깃헙 리포지토리에 README 파일에 테마를 어떻게 적용하는지 나와있다는 점이었다!!!

아 얼마나 든든한가. 나도 다음에 뭐 만들 때 README 파일을 잘 작성해야겠다고 다짐했다. 지금까지 야매로 찾아보던 나에게 공식 문서를 들이민 느낌.. 구세주를 얻은 느낌.. 신뢰가 가지 않는가. 바로 요 테마로 골랐다.

![Untitled](/_publications/240529/Untitled%2014.png)

![Untitled](/_publications/240529/53c89166-6418-4a74-b76d-46185a11be39.png)

[Lucypothesis.github.io](http://Lucypothesis.github.io)에 남아있는 지금까지의 나의 삽질 커밋로그들을 날려버리기 아쉬워서 기존 리포지토리 이름을 was-lucypothesis.github.io로 바꾸고 새로 Lucypothesis.github.io 리포지토리를 만들었다. 여기에 Academic Pages 테마를 적용할 것이다.  

README.md에 있는 Getting Started 보면서 차근차근 따라하기만 하면 된다. 원래 git bash 쓰면서 했는데 이거는 그냥 github 사이트에서 하라는대로 진행했다. “Use this template” 버튼만 누르면 된다!! 아 혁신적이다. 이게 이렇게 쉽게 끝나는 거였다니. _config.yml 파일과 [about.md](http://about.md) 파일을 수정한 결과..

![Untitled](/_publications/240529/Untitled%2015.png)

깃헙블로그가 탄생했다!!!! 감격스럽다. 이 짧고 간결한 과정을 하기 위해 얼마나 삽질을 했던가. 의도한 결과물을 처음으로 딱 봤을 때만의 그 희열이 있다. 결과를 보면 과정이 미화되는 경향이 강하지만.. 이맛에 컴퓨터 한다.

깃헙 블로그를 써야하니 문서 보면서 마크다운 문법도 익히기로 했다. 번외로 블로그 만들면서 README.md에서 ‘md’가 markdown이라는 걸 알게 되었다. 하하.. 기초 없이 모래성 쌓는 느낌을 지울 수 없지만 뭐 어쩌겠는가. 지금부터 하나하나 배워나가야지.. 

### [교훈]

1. 공식 문서가 있으면 **공식 문서를 제일 먼저 읽자.** 다른 사람 블로그는 그 다음이다.
2. 배우는 과정에서는 **GPT 사용을 최소화하자**. 에러코드 그냥 복사 붙여넣기 하지말고 직접 에러코드를 읽고 해결해보려고 노력하자. 그래야 이 프로그램이 어떻게 생겼는지 보고  왜 이렇게 배치했는지 생각하게 되면서 많이 배우게 되는 것 같다. 
3. 귀찮아하지 말고 **영어 잘 읽자.** 해결책은 눈앞에 있다. 단지 영어로 나와있을 뿐….
4. **에러가 나면 일단 캡쳐하자**. 블로그를 쓰려고 보니 그때그때 에러난 부분을 기록해놓지 않은게 아쉽다. 내가 무슨 에러를 마주했고 어떤 걸 시도했고 해결했다면 어떻게 해결했는지 블로그에 기록해놓고 싶다. 시간이 지나면 나도 잊어버리게 되는데 나의 이런 굼벵이 기어가던 시절을 잊고 싶지 않다. 나중에 돌아보면 나도 이런 시절이 있었구나 하고 싶다. 
    
    4-1. **그때그때 기록하자**. 다 한 다음에 기록하려고 하니 너무 많고 순서가 정리가 안 된다. 끊는 타이밍 잘 잡아서 바로바로 기록해놓자. 
    

+추가) 나의 마인드셋

<aside>
💡 if 에러:
	print(‘오히려 좋아 블로그 쓸거리 생겼어’)
else:
	print(’원하는대로 흘러가고 있군.’)

</aside>

다사다난했던 깃헙 블로그 제작 일기 끝~