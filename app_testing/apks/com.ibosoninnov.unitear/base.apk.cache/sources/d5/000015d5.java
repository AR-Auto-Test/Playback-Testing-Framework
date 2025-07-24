package com.google.android.gms.signin.internal;

import android.os.IInterface;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.common.ConnectionResult;
import com.google.android.gms.common.api.Status;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public interface zac extends IInterface {
    void zaa(ConnectionResult connectionResult, zab zabVar);

    void zaa(Status status);

    void zaa(Status status, GoogleSignInAccount googleSignInAccount);

    void zaa(zag zagVar);

    void zaa(zam zamVar);

    void zab(Status status);
}