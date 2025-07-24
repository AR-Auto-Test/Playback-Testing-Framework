package com.ibosoninnov.unitear;

import android.content.Intent;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import b.b.c.h;
import java.util.Objects;

/* loaded from: classes2.dex */
public class MyWbActivity extends h {

    /* loaded from: classes2.dex */
    public class a extends WebChromeClient {
        public a(MyWbActivity myWbActivity) {
        }
    }

    /* loaded from: classes2.dex */
    public class b extends WebViewClient {
        public b() {
        }

        @Override // android.webkit.WebViewClient
        public void onPageFinished(WebView webView, String str) {
            super.onPageFinished(webView, str);
            Objects.requireNonNull(MyWbActivity.this);
            throw null;
        }

        @Override // android.webkit.WebViewClient
        public void onPageStarted(WebView webView, String str, Bitmap bitmap) {
            super.onPageStarted(webView, str, bitmap);
        }

        @Override // android.webkit.WebViewClient
        public boolean shouldOverrideUrlLoading(WebView webView, String str) {
            webView.loadUrl(str);
            return true;
        }
    }

    @Override // b.q.b.d, android.app.Activity
    public void onActivityResult(int i, int i2, Intent intent) {
        if (i == 1) {
        }
    }

    @Override // b.b.c.h, b.q.b.d, android.app.Activity, android.content.ComponentCallbacks
    public void onConfigurationChanged(Configuration configuration) {
        super.onConfigurationChanged(configuration);
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_login);
        WebView webView = (WebView) findViewById(R.id.webView);
        WebView webView2 = new WebView(this);
        webView2.getSettings().setJavaScriptEnabled(true);
        webView2.loadUrl("http://www.script-tutorials.com/demos/199/index.html");
        webView2.setWebViewClient(new b());
        webView2.setWebChromeClient(new a(this));
        setContentView(webView2);
    }
}