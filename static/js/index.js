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
  const heroPlayFallback = document.querySelector("#hero-play-fallback");

  if (heroVideo) {
    let fallbackTimer;

    // iOS Safari evaluates autoplay from the live media properties as well as
    // the HTML attributes. Set both before every playback attempt.
    const prepareHeroVideo = () => {
      heroVideo.defaultMuted = true;
      heroVideo.muted = true;
      heroVideo.playsInline = true;
      heroVideo.setAttribute("muted", "");
      heroVideo.setAttribute("playsinline", "");
      heroVideo.setAttribute("webkit-playsinline", "");
    };

    const hideHeroFallback = () => {
      window.clearTimeout(fallbackTimer);
      if (heroPlayFallback) heroPlayFallback.hidden = true;
    };

    const scheduleHeroFallback = () => {
      window.clearTimeout(fallbackTimer);
      fallbackTimer = window.setTimeout(() => {
        if (heroPlayFallback && heroVideo.paused && !document.hidden) {
          heroPlayFallback.hidden = false;
        }
      }, 900);
    };

    const playHeroVideo = () => {
      prepareHeroVideo();
      if (document.hidden || !heroVideo.paused) {
        if (!heroVideo.paused) hideHeroFallback();
        return;
      }

      let playAttempt;
      try {
        // Keep play() synchronous with click/touchend so Safari recognizes the
        // user's first interaction when automatic playback was initially denied.
        playAttempt = heroVideo.play();
      } catch (error) {
        scheduleHeroFallback();
        return;
      }

      if (playAttempt && typeof playAttempt.then === "function") {
        playAttempt.then(hideHeroFallback).catch(scheduleHeroFallback);
      }
    };

    prepareHeroVideo();
    heroVideo.addEventListener("loadedmetadata", playHeroVideo);
    heroVideo.addEventListener("loadeddata", playHeroVideo);
    heroVideo.addEventListener("canplay", playHeroVideo);
    heroVideo.addEventListener("playing", hideHeroFallback);
    heroVideo.addEventListener("error", scheduleHeroFallback);
    window.addEventListener("pageshow", playHeroVideo);
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) playHeroVideo();
    });

    // WebKit only treats a direct event-handler call as a user gesture.
    document.addEventListener("touchend", playHeroVideo, { passive: true });
    document.addEventListener("click", playHeroVideo);
    document.addEventListener("keydown", playHeroVideo);
    heroPlayFallback?.addEventListener("click", playHeroVideo);

    if ("IntersectionObserver" in window) {
      const heroObserver = new IntersectionObserver((entries) => {
        if (entries.some((entry) => entry.isIntersecting)) playHeroVideo();
      }, { threshold: 0.05 });
      heroObserver.observe(heroVideo);
    }

    playHeroVideo();
  }
});
