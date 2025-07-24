package com.ibosoninnov.unitear;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.PorterDuff;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.format.DateFormat;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.webkit.DownloadListener;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import b.b.c.h;
import c.e.b.fe;
import c.e.b.zd;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.ibosoninnov.unitear.LoginWebviewActivity;
import f.v;
import f.x;
import f.y;
import java.io.File;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.URLDecoder;
import java.util.Date;
import java.util.Objects;

/* loaded from: classes2.dex */
public class LoginWebviewActivity extends h {
    public static final /* synthetic */ int r = 0;
    public WebView s;
    public ImageView t;
    public LinearLayout u;
    public TextView v;
    public TextView w;
    public ValueCallback<Uri[]> x;

    /* loaded from: classes2.dex */
    public class a implements DownloadListener {
        public a() {
        }

        @Override // android.webkit.DownloadListener
        public void onDownloadStart(String str, String str2, String str3, String str4, long j) {
            StringBuilder sb = new StringBuilder();
            sb.append(Environment.DIRECTORY_PICTURES);
            String str5 = File.separator;
            sb.append(str5);
            sb.append("UniteAR");
            sb.append(str5);
            sb.append("Downloads");
            String sb2 = sb.toString();
            File file = new File(sb2);
            if (!file.exists()) {
                file.mkdirs();
            }
            Log.d(LoginWebviewActivity.class.getName(), "webview setDownloadListener " + str);
            if (str.startsWith("data:")) {
                String substring = str.substring(str.indexOf("/") + 1, str.indexOf(";"));
                if (!substring.contains("jpeg") && !substring.contains("jpg") && !substring.contains("png")) {
                    if (substring.contains("svg")) {
                        LoginWebviewActivity loginWebviewActivity = LoginWebviewActivity.this;
                        Objects.requireNonNull(loginWebviewActivity);
                        File file2 = new File(sb2);
                        try {
                            byte[] bytes = URLDecoder.decode(str.substring(str.indexOf(",") + 1), "UTF-8").getBytes();
                            CharSequence format = DateFormat.format("yyyy-MM-dd_hh:mm:ss", new Date());
                            String str6 = "IMG" + ((Object) format) + ".svg";
                            ContentResolver contentResolver = loginWebviewActivity.getContentResolver();
                            ContentValues contentValues = new ContentValues();
                            contentValues.put("_display_name", str6);
                            contentValues.put("mime_type", "image/svg");
                            contentValues.put("relative_path", sb2);
                            Uri insert = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
                            Objects.requireNonNull(insert);
                            OutputStream openOutputStream = contentResolver.openOutputStream(insert);
                            openOutputStream.write(bytes);
                            openOutputStream.close();
                            loginWebviewActivity.v(file2 + "/" + str6);
                            return;
                        } catch (Exception e2) {
                            Log.w("ExternalStorage", "Error writing " + file2, e2);
                            Toast.makeText(loginWebviewActivity.getApplicationContext(), "Failed", 1).show();
                            return;
                        }
                    }
                    return;
                }
                LoginWebviewActivity loginWebviewActivity2 = LoginWebviewActivity.this;
                Objects.requireNonNull(loginWebviewActivity2);
                File file3 = new File(sb2);
                String substring2 = str.substring(str.indexOf("/") + 1, str.indexOf(";"));
                File file4 = new File(file3, System.currentTimeMillis() + "." + substring2);
                try {
                    byte[] decode = Base64.decode(str.substring(str.indexOf(",") + 1), 0);
                    CharSequence format2 = DateFormat.format("yyyy-MM-dd_hh:mm:ss", new Date());
                    String str7 = "IMG" + ((Object) format2) + "." + substring2;
                    ContentResolver contentResolver2 = loginWebviewActivity2.getContentResolver();
                    ContentValues contentValues2 = new ContentValues();
                    contentValues2.put("_display_name", str7);
                    contentValues2.put("mime_type", "image/" + substring2);
                    contentValues2.put("relative_path", sb2);
                    Uri insert2 = contentResolver2.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues2);
                    Objects.requireNonNull(insert2);
                    OutputStream openOutputStream2 = contentResolver2.openOutputStream(insert2);
                    openOutputStream2.write(decode);
                    openOutputStream2.close();
                    loginWebviewActivity2.v(file3 + "/" + str7);
                } catch (Exception e3) {
                    Log.w("ExternalStorage", "Error writing " + file4, e3);
                    Toast.makeText(loginWebviewActivity2.getApplicationContext(), "Failed", 1).show();
                }
            } else if (!str.endsWith("jpeg") && !str.endsWith("jpg") && !str.endsWith("png")) {
                LoginWebviewActivity.this.startActivity(new Intent("android.intent.action.VIEW", Uri.parse(str)));
            } else {
                LoginWebviewActivity loginWebviewActivity3 = LoginWebviewActivity.this;
                int i = LoginWebviewActivity.r;
                Objects.requireNonNull(loginWebviewActivity3);
                y.a aVar = new y.a();
                aVar.d(str);
                ((x) new v().a(aVar.a())).b(new fe(loginWebviewActivity3, sb2));
            }
        }
    }

    /* loaded from: classes2.dex */
    public class b extends WebViewClient {
        public b(LoginWebviewActivity loginWebviewActivity) {
        }

        @Override // android.webkit.WebViewClient
        public void onLoadResource(WebView webView, String str) {
            super.onLoadResource(webView, str);
        }

        @Override // android.webkit.WebViewClient
        public void onPageFinished(WebView webView, String str) {
            Log.e("WebView", "your current url when webpage loading.. finish" + str);
            super.onPageFinished(webView, str);
        }

        @Override // android.webkit.WebViewClient
        public void onPageStarted(WebView webView, String str, Bitmap bitmap) {
            super.onPageStarted(webView, str, bitmap);
            Log.e("WebView", "your current url when webpage loading.." + str);
        }

        @Override // android.webkit.WebViewClient
        public boolean shouldOverrideUrlLoading(WebView webView, String str) {
            PrintStream printStream = System.out;
            printStream.println("when you click on any interlink on webview that time you got url :-" + str);
            return super.shouldOverrideUrlLoading(webView, str);
        }
    }

    @Override // b.q.b.d, android.app.Activity
    public void onActivityResult(int i, int i2, Intent intent) {
        ValueCallback<Uri[]> valueCallback;
        super.onActivityResult(i, i2, intent);
        if (i != 100 || (valueCallback = this.x) == null) {
            return;
        }
        valueCallback.onReceiveValue(WebChromeClient.FileChooserParams.parseResult(i2, intent));
        this.x = null;
    }

    @Override // androidx.activity.ComponentActivity, android.app.Activity
    public void onBackPressed() {
        if (this.s.canGoBack()) {
            this.s.goBack();
        } else {
            this.f41f.b();
        }
    }

    @Override // b.b.c.h, b.q.b.d, androidx.activity.ComponentActivity, b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_login);
        this.t = (ImageView) findViewById(R.id.iv_close);
        this.s = (WebView) findViewById(R.id.webView);
        this.v = (TextView) findViewById(R.id.tv_close);
        this.u = (LinearLayout) findViewById(R.id.ll_close);
        this.w = (TextView) findViewById(R.id.tv_go_to_home);
        ImageView imageView = this.t;
        Context applicationContext = getApplicationContext();
        Object obj = b.j.c.a.f2074a;
        imageView.setColorFilter(applicationContext.getColor(R.color.grey_333333), PorterDuff.Mode.MULTIPLY);
        WebView webView = (WebView) findViewById(R.id.webView);
        this.s = webView;
        webView.setScrollBarStyle(0);
        this.s.getSettings().setLoadsImagesAutomatically(true);
        this.s.getSettings().setJavaScriptEnabled(true);
        this.s.getSettings().setUserAgentString("YourAppName");
        this.s.loadUrl("https://app.unitear.com/login");
        this.s.getSettings().setDomStorageEnabled(true);
        this.s.getSettings().setAllowContentAccess(true);
        this.s.getSettings().setAllowFileAccess(true);
        this.s.setDownloadListener(new a());
        this.s.setWebViewClient(new b(this));
        WebSettings settings = this.s.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setSupportZoom(false);
        settings.setAllowFileAccess(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        this.s.setWebChromeClient(new zd(this));
        this.t.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.xa
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                LoginWebviewActivity loginWebviewActivity = LoginWebviewActivity.this;
                if (loginWebviewActivity.u.getVisibility() == 0) {
                    loginWebviewActivity.u.animate().alpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).setDuration(300L).setListener(new ae(loginWebviewActivity));
                } else {
                    loginWebviewActivity.u.animate().alpha(1.0f).setDuration(300L).setListener(new be(loginWebviewActivity));
                }
            }
        });
        this.v.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.ya
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                LoginWebviewActivity loginWebviewActivity = LoginWebviewActivity.this;
                loginWebviewActivity.u.animate().alpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).setDuration(300L).setListener(new ce(loginWebviewActivity));
                loginWebviewActivity.onBackPressed();
            }
        });
        this.w.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.va
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                LoginWebviewActivity loginWebviewActivity = LoginWebviewActivity.this;
                loginWebviewActivity.u.animate().alpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).setDuration(300L).setListener(new de(loginWebviewActivity));
                if (loginWebviewActivity.s.getUrl().equals("https://app.unitear.com/login")) {
                    loginWebviewActivity.s.loadUrl("https://app.unitear.com/login");
                } else {
                    loginWebviewActivity.s.loadUrl("https://app.unitear.com/editor");
                }
            }
        });
        this.s.setOnClickListener(new View.OnClickListener() { // from class: c.e.b.wa
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                LoginWebviewActivity loginWebviewActivity = LoginWebviewActivity.this;
                if (loginWebviewActivity.u.getVisibility() == 0) {
                    loginWebviewActivity.u.animate().alpha(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD).setDuration(300L).setListener(new ee(loginWebviewActivity));
                }
            }
        });
    }

    public final void v(String str) {
        Toast.makeText(this, getResources().getString(R.string.saved) + " : " + str, 1).show();
    }
}