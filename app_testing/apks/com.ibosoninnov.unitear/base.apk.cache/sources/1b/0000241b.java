package com.google.common.net;

import c.b.a.a.a;
import com.google.common.annotations.Beta;
import com.google.common.annotations.GwtIncompatible;
import com.google.common.base.Preconditions;
import java.net.InetAddress;
import java.text.ParseException;

@Beta
@GwtIncompatible
/* loaded from: classes.dex */
public final class HostSpecifier {
    private final String canonicalForm;

    private HostSpecifier(String str) {
        this.canonicalForm = str;
    }

    public static HostSpecifier from(String str) {
        try {
            return fromValid(str);
        } catch (IllegalArgumentException e2) {
            ParseException parseException = new ParseException(a.q("Invalid host specifier: ", str), 0);
            parseException.initCause(e2);
            throw parseException;
        }
    }

    public static HostSpecifier fromValid(String str) {
        HostAndPort fromString = HostAndPort.fromString(str);
        Preconditions.checkArgument(!fromString.hasPort());
        String host = fromString.getHost();
        InetAddress inetAddress = null;
        try {
            inetAddress = InetAddresses.forString(host);
        } catch (IllegalArgumentException unused) {
        }
        if (inetAddress != null) {
            return new HostSpecifier(InetAddresses.toUriString(inetAddress));
        }
        InternetDomainName from = InternetDomainName.from(host);
        if (from.hasPublicSuffix()) {
            return new HostSpecifier(from.toString());
        }
        throw new IllegalArgumentException(a.q("Domain name does not have a recognized public suffix: ", host));
    }

    public static boolean isValid(String str) {
        try {
            fromValid(str);
            return true;
        } catch (IllegalArgumentException unused) {
            return false;
        }
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj instanceof HostSpecifier) {
            return this.canonicalForm.equals(((HostSpecifier) obj).canonicalForm);
        }
        return false;
    }

    public int hashCode() {
        return this.canonicalForm.hashCode();
    }

    public String toString() {
        return this.canonicalForm;
    }
}