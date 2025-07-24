package com.ibosoninnov.unitear;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ProgressBar;
import b.b.c.h;

/* loaded from: classes2.dex */
public class YoutubeView extends h {
    public WebView r;
    public ProgressBar s;
    public String t;

    /* loaded from: classes2.dex */
    public class a extends WebViewClient {
        public a() {
        }

        @Override // android.webkit.WebViewClient
        public void onPageFinished(WebView webView, String str) {
            YoutubeView.this.s.setVisibility(8);
        }
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    @SuppressLint({"SetJavaScriptEnabled"})
    public void onCreate(Bundle bundle) {
        q().r(1);
        getWindow().setFlags(1024, 1024);
        getWindow().addFlags(128);
        getWindow().addFlags(1536);
        super.onCreate(bundle);
        setContentView(R.layout.activity_youtube_view);
        Bundle extras = getIntent().getExtras();
        if (extras != null && extras.containsKey("youtubeID")) {
            this.t = extras.getString("youtubeID");
        }
        this.r = (WebView) findViewById(R.id.youtube_webview);
        this.s = (ProgressBar) findViewById(R.id.webview_progressbar);
        this.r.getSettings().setJavaScriptEnabled(true);
        this.r.setWebViewClient(new a());
        WebView webView = this.r;
        StringBuilder x = c.b.a.a.a.x("https://app.unitear.com/android-allow-embed.html?url=");
        x.append(this.t);
        webView.loadUrl(x.toString());
    }
}