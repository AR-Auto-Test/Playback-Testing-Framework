package com.google.ar.core;

import android.view.View;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

/* compiled from: InstallActivity.java */
/* loaded from: classes.dex */
public final class p implements View.OnClickListener {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ InstallActivity f5594a;

    /* renamed from: b  reason: collision with root package name */
    private final /* synthetic */ int f5595b;

    public p(InstallActivity installActivity, int i) {
        this.f5595b = i;
        this.f5594a = installActivity;
    }

    @Override // android.view.View.OnClickListener
    public final void onClick(View view) {
        if (this.f5595b != 0) {
            this.f5594a.finishWithFailure(new UnavailableUserDeclinedInstallationException());
            return;
        }
        this.f5594a.animateToSpinner();
        this.f5594a.startInstaller();
    }
}