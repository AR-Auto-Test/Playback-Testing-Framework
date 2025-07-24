package com.ibosoninnov.instanttrackinglib;

import android.content.ClipDescription;
import android.content.Context;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.AttributeSet;
import android.util.Log;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import b.b.h.k;
import b.j.j.y.a;
import b.j.j.y.b;
import b.j.j.y.c;
import b.j.j.y.d;
import b.j.j.y.e;

/* loaded from: classes2.dex */
public class GIFEditText extends k {
    private GIFCommitListener gifCommitListener;

    /* loaded from: classes2.dex */
    public interface GIFCommitListener {
        void onGIFCommit(Uri uri, ClipDescription clipDescription);
    }

    public GIFEditText(Context context) {
        super(context);
    }

    @Override // b.b.h.k, android.widget.TextView, android.view.View
    public InputConnection onCreateInputConnection(EditorInfo editorInfo) {
        String[] stringArray;
        InputConnection cVar;
        InputConnection onCreateInputConnection = super.onCreateInputConnection(editorInfo);
        String[] strArr = {"image/gif"};
        int i = Build.VERSION.SDK_INT;
        if (i >= 25) {
            editorInfo.contentMimeTypes = strArr;
        } else {
            if (editorInfo.extras == null) {
                editorInfo.extras = new Bundle();
            }
            editorInfo.extras.putStringArray("androidx.core.view.inputmethod.EditorInfoCompat.CONTENT_MIME_TYPES", strArr);
            editorInfo.extras.putStringArray("android.support.v13.view.inputmethod.EditorInfoCompat.CONTENT_MIME_TYPES", strArr);
        }
        d dVar = new d() { // from class: com.ibosoninnov.instanttrackinglib.GIFEditText.1
            @Override // b.j.j.y.d
            public boolean onCommitContent(final e eVar, int i2, Bundle bundle) {
                try {
                    if (GIFEditText.this.gifCommitListener != null) {
                        new Thread(new Runnable() { // from class: com.ibosoninnov.instanttrackinglib.GIFEditText.1.1
                            @Override // java.lang.Runnable
                            public void run() {
                                eVar.f2277a.b();
                                GIFEditText.this.gifCommitListener.onGIFCommit(eVar.f2277a.a(), eVar.f2277a.c());
                                eVar.f2277a.d();
                            }
                        }).start();
                        return true;
                    }
                    return true;
                } catch (RuntimeException e2) {
                    Log.e("GIFEditText", "Input connection to GIF selection failed");
                    e2.printStackTrace();
                    return false;
                }
            }
        };
        if (onCreateInputConnection != null) {
            if (editorInfo != null) {
                if (i >= 25) {
                    cVar = new b(onCreateInputConnection, false, dVar);
                } else {
                    if (i >= 25) {
                        stringArray = editorInfo.contentMimeTypes;
                        if (stringArray == null) {
                            stringArray = a.f2274a;
                        }
                    } else {
                        Bundle bundle = editorInfo.extras;
                        if (bundle == null) {
                            stringArray = a.f2274a;
                        } else {
                            String[] stringArray2 = bundle.getStringArray("androidx.core.view.inputmethod.EditorInfoCompat.CONTENT_MIME_TYPES");
                            stringArray = stringArray2 == null ? editorInfo.extras.getStringArray("android.support.v13.view.inputmethod.EditorInfoCompat.CONTENT_MIME_TYPES") : stringArray2;
                            if (stringArray == null) {
                                stringArray = a.f2274a;
                            }
                        }
                    }
                    if (stringArray.length == 0) {
                        return onCreateInputConnection;
                    }
                    cVar = new c(onCreateInputConnection, false, dVar);
                }
                return cVar;
            }
            throw new IllegalArgumentException("editorInfo must be non-null");
        }
        throw new IllegalArgumentException("inputConnection must be non-null");
    }

    public void setGIFCommitListener(GIFCommitListener gIFCommitListener) {
        this.gifCommitListener = gIFCommitListener;
    }

    public GIFEditText(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
    }
}