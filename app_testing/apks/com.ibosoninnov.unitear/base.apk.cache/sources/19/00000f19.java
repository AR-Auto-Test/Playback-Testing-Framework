package com.google.android.gms.common.server.response;

import com.google.android.gms.common.server.response.FastParser;
import java.io.BufferedReader;

/* compiled from: com.google.android.gms:play-services-base@@17.4.0 */
/* loaded from: classes.dex */
public final class zad implements FastParser.zaa<Boolean> {
    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // com.google.android.gms.common.server.response.FastParser.zaa
    public final /* synthetic */ Boolean zaa(FastParser fastParser, BufferedReader bufferedReader) {
        boolean zaa;
        zaa = fastParser.zaa(bufferedReader, false);
        return Boolean.valueOf(zaa);
    }
}