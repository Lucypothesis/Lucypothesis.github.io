---
layout: archive
title: Posts
permalink: /posts/
author_profile: true
---

{% if site.author.googlescholar %}
  You can also find my articles on my Google Scholar profile.
{% endif %}
{% include base_path %}

<div class="posts">
  {% for post in site.posts %}
    {% include archive-single.html %}
  {% endfor %}
</div>
