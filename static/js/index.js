document.addEventListener("DOMContentLoaded", () => {
  const wireExclusivePlayback = (video) => {
    video.addEventListener("play", () => {
      document.querySelectorAll("video").forEach((other) => {
        if (other !== video && !other.paused) other.pause();
      });
    });
  };

  const toggle = document.querySelector(".nav-toggle");
  const links = document.querySelector(".navlinks");

  if (toggle && links) {
    toggle.addEventListener("click", () => {
      const open = links.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });

    links.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        links.classList.remove("is-open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  document.querySelectorAll("video").forEach(wireExclusivePlayback);

  const revealItems = document.querySelectorAll("[data-reveal]");
  if ("IntersectionObserver" in window) {
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.14, rootMargin: "0px 0px -8%" });
    revealItems.forEach((item) => revealObserver.observe(item));
  } else {
    revealItems.forEach((item) => item.classList.add("is-visible"));
  }

  const resetYouTubeMedia = (media) => {
    const launch = media.querySelector(".video-launch");
    media.querySelector("iframe.feature-video")?.remove();
    media.dataset.loaded = "false";
    media.classList.remove("is-loading", "is-playing");
    if (launch) {
      launch.disabled = false;
      launch.removeAttribute("aria-busy");
    }
  };

  document.querySelectorAll(".feature-media[data-youtube-id]").forEach((media) => {
    const launch = media.querySelector(".video-launch");
    if (!launch) return;

    launch.addEventListener("click", () => {
      if (media.dataset.loaded === "true") return;

      document.querySelectorAll(".feature-media[data-youtube-id]").forEach((other) => {
        if (other !== media && other.dataset.loaded === "true") resetYouTubeMedia(other);
      });

      media.dataset.loaded = "true";
      media.classList.add("is-loading");
      launch.disabled = true;
      launch.setAttribute("aria-busy", "true");

      const player = document.createElement("iframe");
      player.className = "feature-video";
      player.title = launch.getAttribute("aria-label") || "DynaConTalk video";
      player.src = `https://www.youtube.com/embed/${encodeURIComponent(media.dataset.youtubeId)}?autoplay=1&rel=0&playsinline=1`;
      player.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
      player.referrerPolicy = "strict-origin-when-cross-origin";
      player.allowFullscreen = true;

      player.addEventListener("load", () => {
        media.classList.remove("is-loading");
        media.classList.add("is-playing");
        launch.removeAttribute("aria-busy");
      }, { once: true });

      media.append(player);
    });
  });

  const heroVideo = document.querySelector("#hero-background");
  if (heroVideo) heroVideo.play().catch(() => {});
});
