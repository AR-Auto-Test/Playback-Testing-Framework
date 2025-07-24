package com.google.android.gms.common.api.internal;

import android.os.DeadObjectException;
import android.os.RemoteException;
import com.google.android.gms.common.api.ApiException;
import com.google.android.gms.common.api.Status;
import com.google.android.gms.common.api.internal.GoogleApiManager;
import com.google.android.gms.tasks.TaskCompletionSource;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public abstract class zae<T> extends zab {
    public final TaskCompletionSource<T> zab;

    public zae(int i, TaskCompletionSource<T> taskCompletionSource) {
        super(i);
        this.zab = taskCompletionSource;
    }

    @Override // com.google.android.gms.common.api.internal.zac
    public void zaa(Status status) {
        this.zab.trySetException(new ApiException(status));
    }

    @Override // com.google.android.gms.common.api.internal.zac
    public void zaa(zaw zawVar, boolean z) {
    }

    @Override // com.google.android.gms.common.api.internal.zac
    public final void zac(GoogleApiManager.zaa<?> zaaVar) {
        Status zab;
        Status zab2;
        try {
            zad(zaaVar);
        } catch (DeadObjectException e2) {
            zab2 = zac.zab(e2);
            zaa(zab2);
            throw e2;
        } catch (RemoteException e3) {
            zab = zac.zab(e3);
            zaa(zab);
        } catch (RuntimeException e4) {
            zaa(e4);
        }
    }

    public abstract void zad(GoogleApiManager.zaa<?> zaaVar);

    @Override // com.google.android.gms.common.api.internal.zac
    public void zaa(Exception exc) {
        this.zab.trySetException(exc);
    }
}