package com.google.ar.core;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException;

/* compiled from: InstallActivity.java */
/* loaded from: classes.dex */
public final class s {

    /* renamed from: a  reason: collision with root package name */
    public boolean f5601a = false;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ InstallActivity f5602b;

    public s(InstallActivity installActivity) {
        this.f5602b = installActivity;
    }

    public final void a(t tVar) {
        boolean z;
        synchronized (this.f5602b) {
            if (this.f5601a) {
                return;
            }
            this.f5602b.lastEvent = tVar;
            t tVar2 = t.ACCEPTED;
            ArCoreApk.UserMessageType userMessageType = ArCoreApk.UserMessageType.APPLICATION;
            ArCoreApk.Availability availability = ArCoreApk.Availability.UNKNOWN_ERROR;
            int ordinal = tVar.ordinal();
            if (ordinal != 0) {
                if (ordinal == 1) {
                    this.f5602b.finishWithFailure(new UnavailableUserDeclinedInstallationException());
                } else if (ordinal == 2) {
                    z = this.f5602b.waitingForCompletion;
                    if (!z && j.a().f5580b) {
                        this.f5602b.closeInstaller();
                    }
                    this.f5602b.finishWithFailure(null);
                }
                this.f5601a = true;
            }
        }
    }

    public final void b(Exception exc) {
        synchronized (this.f5602b) {
            if (this.f5601a) {
                return;
            }
            this.f5601a = true;
            this.f5602b.lastEvent = t.CANCELLED;
            this.f5602b.finishWithFailure(exc);
        }
    }
}