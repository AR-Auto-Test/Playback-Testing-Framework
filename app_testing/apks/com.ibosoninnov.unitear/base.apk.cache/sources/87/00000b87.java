package c.e.b;

import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.net.Uri;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.widget.Toast;
import com.ibosoninnov.unitear.LoginWebviewActivity;

/* compiled from: LoginWebviewActivity.java */
/* loaded from: classes2.dex */
public class zd extends WebChromeClient {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ LoginWebviewActivity f5515a;

    public zd(LoginWebviewActivity loginWebviewActivity) {
        this.f5515a = loginWebviewActivity;
    }

    @Override // android.webkit.WebChromeClient
    public boolean onShowFileChooser(WebView webView, ValueCallback<Uri[]> valueCallback, WebChromeClient.FileChooserParams fileChooserParams) {
        ValueCallback<Uri[]> valueCallback2 = this.f5515a.x;
        if (valueCallback2 != null) {
            valueCallback2.onReceiveValue(null);
            this.f5515a.x = null;
        }
        this.f5515a.x = valueCallback;
        Intent intent = new Intent("android.intent.action.GET_CONTENT");
        intent.addCategory("android.intent.category.OPENABLE");
        intent.setType("*/*");
        try {
            this.f5515a.startActivityForResult(intent, 100);
            return true;
        } catch (ActivityNotFoundException e2) {
            this.f5515a.x = null;
            e2.printStackTrace();
            Toast.makeText(this.f5515a.getApplicationContext(), "Cannot Open File Chooser", 1).show();
            return false;
        }
    }
}